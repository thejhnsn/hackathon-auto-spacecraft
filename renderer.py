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


    def render(self, com_orbit_points, obs_orbit_points, com_current_orbit_points, obs_current_orbit_points, comms):

        """ Renders the initial, final and current orbit """
        self.screen.fill(self.BLACK)

        self.draw_orbit(com_orbit_points, self.WHITE)
        self.draw_orbit(obs_orbit_points, self.RED)
        
        # if thruster_on_off == 1:
        #     color = self.RED
        # else:
        #     color = self.WHITE

        # color = self.RED

        # self.draw_point(r, self.BLUE)

        self.draw_current_orbit(com_current_orbit_points, self.RED)
        self.draw_current_orbit(obs_current_orbit_points, self.BLUE)

        if comms == 1:
            # print(com_current_orbit_points[-1])
            self.draw_connection(com_current_orbit_points[-1], obs_current_orbit_points[-1])
        
        # self.draw_attractor()

        pygame.display.flip()
        pygame.display.update()

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
        pygame.draw.circle(self.screen ,color, center, 10, width = 0)
        pass

    def draw_connection(self, pos1, pos2):
        points1 = pos1  / self.RENDER_DISTANCE * self.SCREEN_HEIGHT  + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]
        points2 = pos2  / self.RENDER_DISTANCE * self.SCREEN_HEIGHT  + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]

        pygame.draw.line(self.screen, self.WHITE, points1[0:2], points2[0:2])

    def draw_attractor(self,  R = 24622):

        """
        Draws sphere with an specified radius and the center of the sphere in the screen
        """

        R_scaled = R / self.RENDER_DISTANCE * self.SCREEN_HEIGHT
        pygame.draw.circle(self.screen, self.WHITE, [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2], R_scaled, width = 0)

    def draw_point(self, r, color):
        point = r / self.RENDER_DISTANCE * self.SCREEN_HEIGHT + [self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2]
        pygame.draw.circle(self.screen ,color, point, 10, width = 0)
                    