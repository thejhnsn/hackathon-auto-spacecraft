import numpy as np
import pygame



class Renderer():
    
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720

    RENDER_DISTANCE = 140000/2#14000

    BLACK = (14,17, 17)
    RED   = (239, 134, 119)
    BLUE  = (130, 182, 217)
    WHITE = (251, 250, 245)


    def __init__(self):
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.screen.fill(self.BLACK)
        self.clock = pygame.time.Clock()


    def render(self, com_orbit_points, obs_orbit_points, com_current_orbit_points, obs_current_orbit_points, comms, commsdata, energy, prop, time):

        """ Renders the initial, final and current orbit """
        self.screen.fill((255, 255,255))

        self.draw_orbit(com_orbit_points, self.BLACK)
        self.draw_orbit(obs_orbit_points, self.RED)
        self.draw_attractor()

        # if thruster_on_off == 1:
        #     color = self.RED
        # else:
        #     color = self.WHITE

        # color = self.RED

        # self.draw_point(r, self.BLUE)

        self.draw_current_orbit(com_current_orbit_points, self.RED)
        self.draw_current_orbit(obs_current_orbit_points, self.BLUE)

            # print(com_current_orbit_points[-1])
        self.draw_connection(com_current_orbit_points[-1], obs_current_orbit_points[-1])
        self.drawbars(commsdata, energy, prop, time)
        # self.draw_attractor()

        pygame.display.flip()
        pygame.display.update()

    def drawbars(self, comms, energy , prop  , time):
        comms = (comms/60000000)
        energy = (energy/15000)
        prop = (prop/2)
        time = time/60
        time = (time/5000)
        print(f"Coms: {comms} Energy: {energy} Prop: {prop} Time: {time}")

        pygame.draw.rect(self.screen, (217, 217, 217), pygame.Rect(50, 10, 410, 30))
        pygame.draw.rect(self.screen, (217, 217, 217), pygame.Rect(50, 60, 410, 30))
        pygame.draw.rect(self.screen, (217, 217, 217), pygame.Rect(50, 110, 410, 30))


        pygame.draw.rect(self.screen, (51, 255, 255), pygame.Rect(55, 12, 400*prop, 22))  #prop
        pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(55, 62, 400*energy, 22)) #energy
        pygame.draw.rect(self.screen, (255, 26, 26), pygame.Rect(55, 112,  400*comms, 22)) #comms




    def draw_orbit(self, positions, color):

        """ Draws orbit given a vector with the positions 

        :param positions: np.array with orbit points
        :param color: Tuple with RGB values. Ex: Red = (255,0, 0)

        """

        points = positions / self.RENDER_DISTANCE * self.SCREEN_HEIGHT + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]
        pygame.draw.aalines(self.screen, color, False, points)

    def draw_current_orbit(self, positions, color):

        """
        Draws orbit given a numpy array with the points to be drawn in order
        """

        points = positions / self.RENDER_DISTANCE * self.SCREEN_HEIGHT  + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]
        center = points[-1:,:][0]
        pygame.draw.aalines(self.screen, self.WHITE, False, points)
        satelite = pygame.image.load('Sateliet.png')
        satelite = pygame.transform.scale(satelite, (36,36))
        self.screen.blit(satelite, center - [18, 18])
        #pygame.draw.circle(self.screen ,color, center, 10, width = 0)
        pass

    def draw_connection(self, pos1, pos2):
        points1 = pos1  / self.RENDER_DISTANCE * self.SCREEN_HEIGHT  + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]
        points2 = pos2  / self.RENDER_DISTANCE * self.SCREEN_HEIGHT  + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]

        pygame.draw.line(self.screen, self.BLACK, points1[0:2], points2[0:2])

    def draw_attractor(self,  R = 12000):

        """
        Draws sphere with an specified radius and the center of the sphere in the screen
        """

        R_scaled = R / self.RENDER_DISTANCE * self.SCREEN_HEIGHT
        neptun = pygame.image.load('Neptun.jpg')
        size = neptun.get_size()
        self.screen.blit(neptun, ((self.SCREEN_WIDTH/2) - size[0]/2, (self.SCREEN_HEIGHT/2) - size[1]/2))

    def draw_point(self, r, color):
        point = r / self.RENDER_DISTANCE * self.SCREEN_HEIGHT + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]
        pygame.draw.circle(self.screen ,color, point, 10, width = 0)
                    