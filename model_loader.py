import trimesh, numpy
from classes import Model

def pack_image(img):

	arr = numpy.array(img)

	packed = (
		(arr[:, :, 0].astype(numpy.uint32) << 16) |
		(arr[:, :, 1].astype(numpy.uint32) << 8)  |
		(arr[:, :, 2].astype(numpy.uint32))
	)

	return packed

def load_scene(path):

	print(f"Loading ({path})...")

	scene = trimesh.load(path)

	models = []

	for node_name in scene.graph.nodes_geometry:

		model = Model()

		transform, name = scene.graph[node_name]
		mesh = scene.geometry[name]

		model.mesh.vertex_normals = mesh.vertex_normals
		model.mesh.face_normals = mesh.face_normals

		model.mesh.vertices = mesh.vertices.astype(numpy.float32)
		model.mesh.faces = mesh.faces.astype(numpy.uint32)

		# GET UVS

		if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
			model.mesh.uvs = mesh.visual.uv.astype(numpy.float32)
		else:
			print(f"Model <{name}> Doesnt Have UVs")
			model.mesh.uvs = numpy.zeros((len(model.mesh.vertices), 2), dtype=numpy.float32)

		# GET TRANSFORMS

		scale, shear, angles, translation, perspective = trimesh.transformations.decompose_matrix(transform)

		model.position = translation.astype(numpy.float32)
		model.rotation = numpy.array(angles, dtype=numpy.float32)
		model.scale = scale.astype(numpy.float32)

		# GET TEXTURE

		if hasattr(mesh.visual, "material"):

			material = mesh.visual.material

			if hasattr(material, "baseColorTexture"):

				model.texture = pack_image(material.baseColorTexture)

			else:

				print(f"Model <{name}> Doesnt Have Texture")

		else:

			print(f"Model <{name}> Doesnt Have Material")

		models.append(model)

	return models