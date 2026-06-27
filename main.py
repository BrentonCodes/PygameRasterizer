import pygame
import numpy
import model_loader
import renderer
from classes import Camera, PointLight

models = model_loader.load_scene("scene.glb")

RESOLUTION = (1280,960)
TARGET_FPS = 0

speed = 8
sense = 0.0015

window = pygame.display.set_mode(RESOLUTION)

framebuffer = numpy.zeros(RESOLUTION, dtype=numpy.uint32)
depthbuffer = numpy.zeros(RESOLUTION, dtype=numpy.float32)

FOV = numpy.radians(70)
camera = Camera()

test_light = PointLight()
test_light.color = (255,0,0)
test_light.intensity = 0.005

test_light2 = PointLight()
test_light2.color = (0,255,0)
test_light2.intensity = 0.0025

camera.position[2] = -15
camera.position[1] = 5
camera.rotation[0] = .25

camera.f = (RESOLUTION[0] / 2) / numpy.tan(FOV / 2)

pygame.event.set_grab(True)
pygame.mouse.set_visible(False)

clock = pygame.time.Clock()

running = True

cx, cy = 0, 0

r = 5

theta = 0

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

	theta += dt
	
	models[3].position[0] = cx + r * numpy.cos(theta)
	models[3].position[2] = cy + r * numpy.sin(theta)

	test_light.position = models[3].position
	test_light2.position = camera.position

	for model in models:

		renderer.render_model(framebuffer,depthbuffer,model,camera,[test_light,test_light2])

	pygame.surfarray.blit_array(window, framebuffer)

	pygame.display.flip()

	pygame.display.set_caption(f"FPS: {clock.get_fps():.2f}")