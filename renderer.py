import numpy
from numba import njit

@njit(inline='always')
def apply_lighting_to_packed(color, intensity, light_color=(0,0,0)):

	r = (color >> 16) & 0xFF
	g = (color >> 8) & 0xFF
	b = color & 0xFF
	
	lit_r = min(255, int(r * light_color[0]))
	lit_g = min(255, int(g * light_color[1]))
	lit_b = min(255, int(b * light_color[2]))
	
	return (lit_r << 16) | (lit_g << 8) | lit_b

@njit(inline='always')
def edge(x0, y0, x1, y1, x2, y2):
	return (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)

@njit
def rasterize_triangle(fb, zb, tx, p0, p1, p2, light_intensity, light_color):

	x0, y0, z0_view, u0, v0, _, _ = p0
	x1, y1, z1_view, u1, v1, _, _ = p1
	x2, y2, z2_view, u2, v2, _, _ = p2

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
					fb[x, y] = apply_lighting_to_packed(color,light_intensity,light_color)

			w0 += B0
			w1 += B1
			w2 += B2
			
		w0_row += A0
		w1_row += A1
		w2_row += A2

@njit
def transfrom_normals(normals,mesh_rotation,camera_rotation):

	num_norms = normals.shape[0]

	out = numpy.empty((num_norms, 3), dtype=numpy.float32)

	rx, ry, rz = mesh_rotation[0], mesh_rotation[1], mesh_rotation[2]
	cos_my, sin_my = numpy.cos(ry), numpy.sin(ry)
	cos_mx, sin_mx = numpy.cos(rx), numpy.sin(rx)
	cos_mz, sin_mz = numpy.cos(rz), numpy.sin(rz)

	crx, cry, crz = camera_rotation[0], camera_rotation[1], camera_rotation[2]
	cos_cy, sin_cy = numpy.cos(-cry), numpy.sin(-cry)
	cos_cx, sin_cx = numpy.cos(-crx), numpy.sin(-crx)
	cos_cz, sin_cz = numpy.cos(-crz), numpy.sin(-crz)

	for i in range(num_norms):

		nx = normals[i, 0]
		ny = normals[i, 1]
		nz = normals[i, 2]

		# Mesh rotation
		if ry != 0:
			nx, nz = nx * cos_my + nz * sin_my, -nx * sin_my + nz * cos_my
		if rx != 0:
			ny, nz = ny * cos_mx - nz * sin_mx, ny * sin_mx + nz * cos_mx
		if rz != 0:
			nx, ny = nx * cos_mz - ny * sin_mz, nx * sin_mz + ny * cos_mz

		# Camera rotation
		if cry != 0:
			nx, nz = nx * cos_cy + nz * sin_cy, -nx * sin_cy + nz * cos_cy
		if crx != 0:
			ny, nz = ny * cos_cx - nz * sin_cx, ny * sin_cx + nz * cos_cx
		if crz != 0:
			nx, ny = nx * cos_cz - ny * sin_cz, nx * sin_cz + ny * cos_cz

		# Normalize
		length = numpy.sqrt(nx*nx + ny*ny + nz*nz)
		nx /= length
		ny /= length
		nz /= length

		out[i, 0] = nx
		out[i, 1] = ny
		out[i, 2] = nz

	return out

@njit
def transform_mesh(vertices, uvs, mesh_position, mesh_rotation, mesh_scale, camera_position, camera_rotation, focal_length, width, height):
	
	num_verts = vertices.shape[0]

	out = numpy.empty((num_verts, 7), dtype=numpy.float32)

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
		out[i, 5] = tx
		out[i, 6] = ty

	return out

@njit
def draw_mesh(framebuffer, depthbuffer, texture, points, faces, face_normals, light_positions, light_intensities, light_colors):
	num_lights = light_positions.shape[0]

	for i in range(faces.shape[0]):
		f = faces[i]

		v0 = points[f[0]]
		v1 = points[f[1]]
		v2 = points[f[2]]

		if v0[2] > 0 and v1[2] > 0 and v2[2] > 0:
			cx = (v0[5] + v1[5] + v2[5]) / 3.0
			cy = (v0[6] + v1[6] + v2[6]) / 3.0
			cz = (v0[2] + v1[2] + v2[2]) / 3.0

			nx = face_normals[i, 0]
			ny = face_normals[i, 1]
			nz = face_normals[i, 2]

			sum_intensity = 0.0

			ambient = 0

			sum_light_r = ambient
			sum_light_g = ambient
			sum_light_b = ambient

			for j in range(num_lights):

				lx = light_positions[j, 0] - cx
				ly = light_positions[j, 1] - cy
				lz = light_positions[j, 2] - cz

				dist = numpy.sqrt(lx*lx + ly*ly + lz*lz)

				if dist > 0:
					lx /= dist; ly /= dist; lz /= dist
				
				attenuation = 1.0 / (1.0 + 0.05 * dist + 0.005 * (dist * dist))

				dot = nx*lx + ny*ly + nz*lz

				intensity = max(0.0, dot * light_intensities[j]) * attenuation

				sum_light_r += light_colors[j, 0] * intensity
				sum_light_g += light_colors[j, 1] * intensity
				sum_light_b += light_colors[j, 2] * intensity

			color = numpy.zeros(3, dtype=numpy.float32)

			color[0] = sum_light_r
			color[1] = sum_light_g
			color[2] = sum_light_b

			rasterize_triangle(framebuffer, depthbuffer, texture, v0, v1, v2, sum_intensity, color)

def render_model(framebuffer, depthbuffer, model, camera, lights):
    w, h = framebuffer.shape

    verts = model.mesh.vertices
    faces = model.mesh.faces
    uvs = model.mesh.uvs

    points = transform_mesh(verts, uvs, model.position, model.rotation, model.scale, camera.position, camera.rotation, camera.f, w, h)
    face_normals = transfrom_normals(model.mesh.face_normals, model.rotation, camera.rotation)

    crx, cry, crz = camera.rotation[0], camera.rotation[1], camera.rotation[2]
    cos_cy, sin_cy = numpy.cos(-cry), numpy.sin(-cry)
    cos_cx, sin_cx = numpy.cos(-crx), numpy.sin(-crx)
    cos_cz, sin_cz = numpy.cos(-crz), numpy.sin(-crz)

    num_lights = len(lights)
    light_positions = numpy.empty((num_lights, 3), dtype=numpy.float32)
    light_intensities = numpy.empty(num_lights, dtype=numpy.float32)
    light_colors = numpy.empty((num_lights, 3), dtype=numpy.float32)

    for i in range(num_lights):
        lx = lights[i].position[0] - camera.position[0]
        ly = lights[i].position[1] - camera.position[1]
        lz = lights[i].position[2] - camera.position[2]

        if cry != 0:
            lx, lz = lx * cos_cy + lz * sin_cy, -lx * sin_cy + lz * cos_cy
        if crx != 0:
            ly, lz = ly * cos_cx - lz * sin_cx, ly * sin_cx + lz * cos_cx
        if crz != 0:
            lx, ly = lx * cos_cz - ly * sin_cz, lx * sin_cz + ly * cos_cz

        light_positions[i, 0] = lx
        light_positions[i, 1] = ly
        light_positions[i, 2] = lz
        light_intensities[i] = lights[i].intensity
        
        light_colors[i, 0] = lights[i].color[0]
        light_colors[i, 1] = lights[i].color[1]
        light_colors[i, 2] = lights[i].color[2]

    draw_mesh(framebuffer, depthbuffer, model.texture, points, faces, face_normals, light_positions, light_intensities, light_colors)