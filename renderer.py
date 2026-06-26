import numpy
from numba import njit

@njit(inline='always')
def edge(x0, y0, x1, y1, x2, y2):
	return (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)

@njit
def rasterize_triangle(fb, zb, tx, p0, p1, p2):

	x0, y0, z0_view, u0, v0 = p0
	x1, y1, z1_view, u1, v1 = p1
	x2, y2, z2_view, u2, v2 = p2

	area = edge(x0, y0, x1, y1, x2, y2)

	if area >= 0:
		return

	w, h = fb.shape
	th, tw = tx.shape[:2] 

	minx = max(0,   int(min(x0, x1, x2)))
	maxx = min(w-1, int(max(x0, x1, x2)))
	miny = max(0,   int(min(y0, y1, y2)))
	maxy = min(h-1, int(max(y0, y1, y2)))

	inv_area = 1.0 / area

	A0 = (y2 - y1) * inv_area; B0 = (x1 - x2) * inv_area
	A1 = (y0 - y2) * inv_area; B1 = (x2 - x0) * inv_area
	A2 = (y1 - y0) * inv_area; B2 = (x0 - x1) * inv_area

	w0_row = edge(x1, y1, x2, y2, minx, miny) * inv_area
	w1_row = edge(x2, y2, x0, y0, minx, miny) * inv_area
	w2_row = edge(x0, y0, x1, y1, minx, miny) * inv_area

	iz0 = 1.0 / z0_view
	iz1 = 1.0 / z1_view
	iz2 = 1.0 / z2_view

	uz0 = u0 * iz0; vz0 = v0 * iz0
	uz1 = u1 * iz1; vz1 = v1 * iz1
	uz2 = u2 * iz2; vz2 = v2 * iz2

	for x in range(minx, maxx + 1):

		w0 = w0_row
		w1 = w1_row
		w2 = w2_row
		
		for y in range(miny, maxy + 1):
			
			if w0 >= 0 and w1 >= 0 and w2 >= 0:

				iz_interp = w0 * iz0 + w1 * iz1 + w2 * iz2

				if iz_interp > zb[x, y]:

					uz_interp = w0 * uz0 + w1 * uz1 + w2 * uz2
					vz_interp = w0 * vz0 + w1 * vz1 + w2 * vz2

					u = uz_interp / iz_interp
					v = vz_interp / iz_interp

					tex_x = int(u * tw) % tw
					tex_y = int(v * th) % th

					color = tx[tex_y, tex_x]

					zb[x, y] = iz_interp
					fb[x, y] = color

			w0 += B0
			w1 += B1
			w2 += B2
			
		w0_row += A0
		w1_row += A1
		w2_row += A2

@njit
def transform_mesh(vertices, uvs, mesh_position, mesh_rotation, mesh_scale, camera_position, camera_rotation, focal_length, width, height):
	
	num_verts = vertices.shape[0]

	out = numpy.empty((num_verts, 5), dtype=numpy.float32)

	rx, ry, rz = mesh_rotation[0], mesh_rotation[1], mesh_rotation[2]
	cos_my, sin_my = numpy.cos(ry), numpy.sin(ry)
	cos_mx, sin_mx = numpy.cos(rx), numpy.sin(rx)
	cos_mz, sin_mz = numpy.cos(rz), numpy.sin(rz)

	crx, cry, crz = camera_rotation[0], camera_rotation[1], camera_rotation[2]
	cos_cy, sin_cy = numpy.cos(-cry), numpy.sin(-cry)
	cos_cx, sin_cx = numpy.cos(-crx), numpy.sin(-crx)
	cos_cz, sin_cz = numpy.cos(-crz), numpy.sin(-crz)
	
	eps = 1e-6
	half_w = width * 0.5
	half_h = height * 0.5

	for i in range(num_verts):

		x = vertices[i, 0]
		y = vertices[i, 1]
		z = vertices[i, 2]

		# SCALING
		sx = x * mesh_scale[0]
		sy = y * mesh_scale[1]
		sz = z * mesh_scale[2]

		# MESH ROTATION
		if ry != 0:
			sx, sz = sx * cos_my + sz * sin_my, -sx * sin_my + sz * cos_my
		if rx != 0:
			sy, sz = sy * cos_mx - sz * sin_mx, sy * sin_mx + sz * cos_mx
		if rz != 0:
			sx, sy = sx * cos_mz - sy * sin_mz, sx * sin_mz + sy * cos_mz

		# TRANSLATION
		tx = sx + mesh_position[0] - camera_position[0]
		ty = sy + mesh_position[1] - camera_position[1]
		tz = sz + mesh_position[2] - camera_position[2]

		# CAMERA ROTATION
		if cry != 0:
			tx, tz = tx * cos_cy + tz * sin_cy, -tx * sin_cy + tz * cos_cy
		if crx != 0:
			ty, tz = ty * cos_cx - tz * sin_cx, ty * sin_cx + tz * cos_cx
		if crz != 0:
			tx, ty = tx * cos_cz - ty * sin_cz, tx * sin_cz + ty * cos_cz

		# PROJECTION
		tz_safe = eps if numpy.abs(tz) < eps else tz

		px = (tx / tz_safe) * focal_length + half_w
		py = -(ty / tz_safe) * focal_length + half_h

		out[i, 0] = px
		out[i, 1] = py
		out[i, 2] = tz
		out[i, 3] = uvs[i, 0]
		out[i, 4] = uvs[i, 1]

	return out

@njit
def draw_mesh(framebuffer, depthbuffer, texture, points, faces):
	
	for i in range(faces.shape[0]):

		f = faces[i]

		v0 = points[f[0]]
		v1 = points[f[1]]
		v2 = points[f[2]]

		if v0[2] > 0 and v1[2] > 0 and v2[2] > 0:

			rasterize_triangle(framebuffer,depthbuffer,texture,v0,v1,v2)

def render_model(framebuffer, depthbuffer, model, camera):

	w,h = framebuffer.shape

	verts = model.mesh.vertices
	faces = model.mesh.faces
	uvs = model.mesh.uvs

	points = transform_mesh(verts, uvs, model.position, model.rotation, model.scale, camera.position, camera.rotation, camera.f, w, h)

	draw_mesh(framebuffer, depthbuffer, model.texture, points, faces)
