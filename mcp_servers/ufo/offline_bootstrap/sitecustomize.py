"""Fail closed on external sockets in UFO offline workers, including children."""
import ipaddress
import os
import socket

if os.environ.get('UFO_OFFLINE') == '1':
    _connect = socket.socket.connect
    _connect_ex = socket.socket.connect_ex

    def _check(sock, address):
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return
        host = address[0]
        if host == 'localhost':
            return
        try:
            if ipaddress.ip_address(host).is_loopback:
                return
        except ValueError:
            pass
        raise OSError(f'UFO offline mode blocked network connection to {host}; provide local model weights')

    def _offline_connect(self, address):
        _check(self, address)
        return _connect(self, address)

    def _offline_connect_ex(self, address):
        _check(self, address)
        return _connect_ex(self, address)

    socket.socket.connect = _offline_connect
    socket.socket.connect_ex = _offline_connect_ex
