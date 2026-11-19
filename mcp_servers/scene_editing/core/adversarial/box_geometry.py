"""Geometry for measured vehicle bodies in the simulation frame."""

import math


def corners(box):
    x,y,length,width,yaw=box
    if not all(math.isfinite(v) for v in box) or min(length,width)<=0:
        raise ValueError('Invalid measured vehicle box')
    c,s=math.cos(yaw),math.sin(yaw)
    return [(x+c*dx-s*dy,y+s*dx+c*dy)
            for dx,dy in ((-length/2,-width/2),(length/2,-width/2),
                          (length/2,width/2),(-length/2,width/2))]


def point_segment(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    length=dx*dx+dy*dy
    t=0.0 if length==0 else max(0.0,min(1.0,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length))
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


def body_contact(first,second):
    a,b=corners(first),corners(second)
    overlap=True
    for yaw in (first[-1],second[-1]):
        for angle in (yaw,yaw+math.pi/2):
            axis=math.cos(angle),math.sin(angle)
            pa=[x*axis[0]+y*axis[1] for x,y in a]
            pb=[x*axis[0]+y*axis[1] for x,y in b]
            if min(max(pa),max(pb))-max(min(pa),min(pb)) < -1e-9:
                overlap=False
    if overlap:return True,0.0
    distances=[point_segment(p,poly[i],poly[(i+1)%4])
               for points,poly in ((a,b),(b,a)) for p in points for i in range(4)]
    return False,min(distances)
