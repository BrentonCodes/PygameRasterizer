import pygame
import numpy
import model_loader
import renderer
from classes import Camera

models = model_loader.load_scene("test_scene.glb")

RESOLUTION = (640,360)
TARGET_FPS = 0

speed = 5.5
sense = 0.0015

window = pygame.display.set_mode(RESOLUTION)

framebuffer = numpy.zeros(RESOLUTION, dtype=numpy.uint32)
depthbuffer = numpy.zeros(RESOLUTION, dtype=numpy.float32)

FOV = numpy.radians(70)
camera = Camera()
camera.f = (RESOLUTION[0] / 2) / numpy.tan(FOV / 2)

pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

clock = pygame.time.Clock()

running = True

while running:

	dt = clock.tick(TARGET_FPS) / 1000

	for event in pygame.event.get():

		if event.type == pygame.QUIT:

			running = False

		if event.type == pygame.KEYDOWN:

			if event.key == pygame.K_ESCAPE:

				running = False

	mx, my = pygame.mouse.get_rel()

	camera.rotation[1] += mx * sense
	camera.rotation[0] += my * sense
	camera.rotation[0] = max(-numpy.pi/2, min(numpy.pi/2, camera.rotation[0]))

	keys = pygame.key.get_pressed()
	
	if keys[pygame.K_w]:
		camera.position[0] += numpy.sin(camera.rotation[1]) * speed * dt
		camera.position[2] += numpy.cos(camera.rotation[1]) * speed * dt

	if keys[pygame.K_s]:
		camera.position[0] -= numpy.sin(camera.rotation[1]) * speed * dt
		camera.position[2] -= numpy.cos(camera.rotation[1]) * speed * dt

	if keys[pygame.K_a]:
		camera.position[0] -= numpy.cos(camera.rotation[1]) * speed * dt
		camera.position[2] += numpy.sin(camera.rotation[1]) * speed * dt

	if keys[pygame.K_d]:
		camera.position[0] += numpy.cos(camera.rotation[1]) * speed * dt
		camera.position[2] -= numpy.sin(camera.rotation[1]) * speed * dt

	if keys[pygame.K_SPACE]:
		camera.position[1] += speed * dt

	if keys[pygame.K_LSHIFT]:
		camera.position[1] -= speed * dt

	framebuffer.fill(0)
	depthbuffer.fill(0)

	models[0].rotation[1] += 1 * dt

	for model in models:

		renderer.render_model(framebuffer,depthbuffer,model,camera)

	pygame.surfarray.blit_array(window, framebuffer)

	pygame.display.flip()

	pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")