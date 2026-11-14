const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[char]));

function mediaCard(item, extra) {
  const title = escapeHtml(item.title || (item.kind === 'video' ? '视频产物' : '图片产物'));
  const note = item.note ? `<small>${escapeHtml(item.note)}</small>` : '';
  const src = escapeHtml(item.src);
  if (item.kind === 'video') {
    return `<figure class="artifact-media artifact-video" data-video-src="${src}" data-state="idle">`
      + `<div class="artifact-video-frame">`
      + `<button class="artifact-video-load" type="button" data-load-video>`
      + `<span class="artifact-play" aria-hidden="true"></span><span data-load-label>点击播放视频</span>`
      + `<small>按需加载，不预先下载</small></button>`
      + `<video controls playsinline preload="none" data-artifact-video hidden></video>`
      + `<div class="artifact-media-status" role="status" hidden><span data-media-message></span>`
      + `<button type="button" data-retry-video hidden>重试</button></div>`
      + `</div><figcaption><span>${title}</span><a href="${src}" target="_blank" rel="noopener">打开视频</a>${note}</figcaption>`
      + `</figure>`;
  }
  return `<figure class="artifact-media artifact-image" data-state="idle">`
    + `<div class="artifact-image-frame">`
    + `<a href="${src}" target="_blank" rel="noopener" title="打开原图">`
    + `<img data-artifact-image data-src="${src}"${extra ? '' : ` src="${src}"`} loading="lazy" decoding="async" alt="${title}"></a>`
    + `<div class="artifact-media-status" role="status" hidden><span>图片加载失败</span>`
    + `<button type="button" data-retry-image>重试</button></div></div>`
    + `<figcaption><span>${title}</span><a href="${src}" target="_blank" rel="noopener">查看原图</a>${note}</figcaption>`
    + `</figure>`;
}

export function artifactPreviewHtml(media) {
  const items = (Array.isArray(media) ? media : []).filter(item =>
    item && ['image', 'video'].includes(item.kind) && typeof item.src === 'string' && item.src);
  if (!items.length) return '';
  const selected = items.filter(item => item.primary);
  const primary = selected.length ? selected : items.slice(0, 1);
  const extras = items.filter(item => !primary.includes(item));
  return `<section class="artifact-preview" aria-label="产物预览">`
    + `<div class="artifact-preview-heading"><b>产物预览</b><span>${items.length} 项</span></div>`
    + `<div class="artifact-preview-list">${primary.map(item => mediaCard(item, false)).join('')}</div>`
    + (extras.length ? `<details class="artifact-preview-more"><summary>查看更多产物（${extras.length}）</summary>`
      + `<div class="artifact-preview-list">${extras.map(item => mediaCard(item, true)).join('')}</div></details>` : '')
    + `</section>`;
}

function retryUrl(src, attempt) {
  return src + (src.includes('?') ? '&' : '?') + '_retry=' + attempt;
}

export function bindArtifactPreview(root) {
  root.querySelectorAll('.artifact-preview-more').forEach(details => {
    const loadImages = () => {
      if (details.open) details.querySelectorAll('img[data-artifact-image]').forEach(img => {
        if (!img.getAttribute('src')) img.src = img.dataset.src;
      });
    };
    details.addEventListener('toggle', loadImages);
    loadImages();
  });

  root.querySelectorAll('img[data-artifact-image]').forEach(img => {
    const frame = img.closest('.artifact-image-frame');
    const status = frame.querySelector('.artifact-media-status');
    const retry = frame.querySelector('[data-retry-image]');
    let attempt = 0;
    img.addEventListener('load', () => {
      frame.dataset.state = 'ready';
      status.hidden = true;
    });
    img.addEventListener('error', () => {
      if (!img.getAttribute('src')) return;
      frame.dataset.state = 'error';
      status.hidden = false;
    });
    retry.addEventListener('click', () => {
      attempt += 1;
      frame.dataset.state = 'loading';
      status.hidden = true;
      img.src = retryUrl(img.dataset.src, attempt);
    });
  });

  root.querySelectorAll('.artifact-video').forEach(card => {
    const video = card.querySelector('video');
    const load = card.querySelector('[data-load-video]');
    const loadLabel = card.querySelector('[data-load-label]');
    const status = card.querySelector('.artifact-media-status');
    const message = card.querySelector('[data-media-message]');
    const retry = card.querySelector('[data-retry-video]');
    let attempt = 0;
    let slowTimer = null;

    const showStatus = (text, canRetry = false) => {
      message.textContent = text;
      retry.hidden = !canRetry;
      status.hidden = false;
    };
    const clearSlowTimer = () => {
      if (slowTimer) clearTimeout(slowTimer);
      slowTimer = null;
    };
    const activate = () => {
      clearSlowTimer();
      card.dataset.state = 'loading';
      load.hidden = true;
      video.hidden = false;
      showStatus('正在加载视频…');
      const src = attempt ? retryUrl(card.dataset.videoSrc, attempt) : card.dataset.videoSrc;
      if (video.getAttribute('src') !== src) video.src = src;
      video.load();
      video.play().catch(() => {});
      slowTimer = setTimeout(() => {
        if (video.readyState < 2) showStatus('加载时间较长，可重试或打开视频', true);
      }, 20000);
    };

    load.addEventListener('click', activate);
    retry.addEventListener('click', () => {
      attempt += 1;
      video.pause();
      video.removeAttribute('src');
      video.load();
      activate();
    });
    video.addEventListener('play', () => {
      (root.ownerDocument || document).querySelectorAll('video[data-artifact-video]').forEach(other => {
        if (other !== video && !other.paused) other.pause();
      });
    });
    video.addEventListener('waiting', () => showStatus('正在缓冲…'));
    video.addEventListener('canplay', () => {
      clearSlowTimer();
      card.dataset.state = 'ready';
      status.hidden = true;
      loadLabel.textContent = '重新播放视频';
    });
    video.addEventListener('error', () => {
      if (!video.getAttribute('src')) return;
      clearSlowTimer();
      card.dataset.state = 'error';
      video.hidden = true;
      load.hidden = false;
      loadLabel.textContent = '重新加载视频';
      showStatus('视频加载失败，请重试或打开视频', true);
    });
  });
}

export function createArtifactPreview(media) {
  const html = artifactPreviewHtml(media);
  if (!html) return null;
  const template = document.createElement('template');
  template.innerHTML = html.trim();
  const root = template.content.firstElementChild;
  bindArtifactPreview(root);
  return root;
}

export function unloadArtifactPreview(root) {
  root.querySelectorAll('video[data-artifact-video]').forEach(video => {
    video.pause();
    video.removeAttribute('src');
    video.load();
  });
}
