import numpy

class Mesh:

	def __init__(self):

		self.vertices = []
		self.faces = []
		self.uvs = []
		self.vertex_normals = []
		self.face_normals = []

class Model:

	def __init__(self):

		self.mesh = Mesh()

		self.rotation = numpy.zeros(3, dtype=numpy.float32)
		self.position = numpy.zeros(3, dtype=numpy.float32)
		self.scale = numpy.zeros(3, dtype=numpy.float32)

		self.texture = numpy.array([[0xFFFFFF]], dtype=numpy.uint32)

class Camera:

	def __init__(self):

		self.rotation = numpy.zeros(3, dtype=numpy.float32)
		self.position = numpy.zeros(3, dtype=numpy.float32)

		self.f = 0

class PointLight:

	def __init__(self):

		self.position = numpy.zeros(3, dtype=numpy.float32)
		self.color = numpy.zeros(3, dtype=numpy.float32)
		self.intensity = 1