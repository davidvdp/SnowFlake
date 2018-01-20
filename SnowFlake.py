import os
import argparse
import  logging
from time import clock

import numpy as np
import cv2

class Particles(object):
    def __init__(self, nr_particles: int, region_width: int, region_height: int, background = 0, particle = 255):
        """
        Generates starting particles.
        :param nr_particles: Amount of particles to generate
        :param region_width: Height of region in which they should be generated in
        :param region_height: Width of region in which they should be generated in
        :param background: background value
        :param particle: foreground value
        """
        self.__background = 0
        self.__particle = 255
        self.__nr_particles = nr_particles
        self.__region_width = region_width
        self.__region_height = region_height
        self.__particles = None
        self.__generate_particles(self.__nr_particles, region_width, region_height)

    @property
    def particles(self):
        return self.__particles

    @particles.setter
    def particles(self, particles):
        self.__particles = particles

    def __generate_particles(self, nr_particles, region_width, region_height):
        if  self.__particles == None:
            self.__particles = np.zeros(shape=(nr_particles, 2), dtype=np.int16)
            self.__particles[:, 0] = np.random.randint(0, region_height, size=self.__particles[:, 0].shape)
            self.__particles[:, 1] = np.random.randint(0, region_width, size=self.__particles[:, 1].shape)
        else:
            raise NotImplementedError

    def update_canvas(self, canvas, clear=True):
        if clear:
            canvas[:,:] = 0
        for particle in self.__particles:
            canvas[particle[0], particle[1]] = self.__particle
        return canvas
    
    def move(self):
        movement_arr = np.random.randint(-1, 2, (self.__particles.shape[0],self.__particles.shape[1]), dtype=np.int16)
        self.__particles += movement_arr
        self.__particles[self.__particles < 0] = 0
        self.__particles[:, 0][self.__particles[:, 0] >= self.__region_height] = self.__region_height - 1
        self.__particles[:, 1][self.__particles[:, 1] >= self.__region_width] = self.__region_width - 1

    def get_particle_count(self):
        return len(self.__particles)

class SnowFlake(object):
    def __init__(self,region_width: int, region_height: int, snowflake_color = 128):
        self.__snow_flake = np.array([[[region_height//2, region_width//2]]])
        self.__snow_flake_color = snowflake_color
'''
    def accumulate_particles(self, particles: Particles):
        particles_arr = particles.particles
        new_flakes = None
        for snow_flake_part in self.__snow_flake:
            norm = np.linalg.norm(snow_flake_part - particles_arr, axis=1)
            mask = np.ones(shape=norm.shape, dtype=np.bool)
            mask[norm<2]=False
            if len(mask[mask==False]) > 0:
                if new_flakes is None:
                    new_flakes = particles_arr[mask==False]
                else:
                    np.concatenate((new_flakes, particles_arr[mask == False]))
            particles_arr = particles_arr[mask]

        if new_flakes is not None:
            self.__snow_flake = np.concatenate((self.__snow_flake,new_flakes))
            particles.particles = particles_arr
            '''

    def update_canvas(self, canvas, clear=False):
        if clear:
            canvas[:,:] = 0
        for particle in self.__snow_flake:
            canvas[particle[0], particle[1]] = self.__snow_flake_color
        return canvas

def create_canvas(width: int, height: int) -> np.array:
    return np.zeros(shape=(height, width), dtype=np.uint8)


def main():
    #parse arguments
    parser = argparse.ArgumentParser(description='Generates a snowflake.')
    parser.add_argument('-p' ,'--nr-particles', help="Number of particles floating around.", default=100, type=int)
    parser.add_argument('--height', help="Height of canvas in pixels.", default=400, type=int)
    parser.add_argument('--width', help="Width of canvas in pixels.", default=640, type=int)
    parser.add_argument('--show-int', help="Show interval", default=10, type=int)

    args = parser.parse_args()

    # set logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s;%(levelname)s;%(message)s')

    #initialize variables
    canvas_width = args.width
    canvas_height = args.height
    nr_particles = args.nr_particles
    show_interval = args.show_int
    show_cnt = show_interval
    step = 0

    #generate canvas
    canvas = create_canvas(canvas_width, canvas_height)

    #generate paricles
    particles = Particles(nr_particles, canvas_width, canvas_height)

    times = []

    frame_rate_time = clock()
    snowflake = SnowFlake(canvas_width, canvas_height)

    # let the particles move
    while True:
        step += 1
        show_cnt += 1
        start = clock()
        particles.move()
        snowflake.accumulate_particles(particles)

        timed = clock() - start
        times.append(timed)

        if show_cnt > show_interval:
            now = clock()
            frame_rate = 1/(now - frame_rate_time)
            frame_rate_time = now

            show_cnt = 0

            average_time = np.mean(np.array(times))
            times = []

            mean = np.mean(canvas)

            canvas = particles.update_canvas(canvas)
            canvas = snowflake.update_canvas(canvas)
            logging.info("Average %.6f seconds per move. %d particles remaining, mean: %.2f @%.3f fps" % (
                average_time,
                particles.get_particle_count(),
                mean,
                frame_rate
            ))

            cv2.imshow("SnowFlake", canvas)
            key = cv2.waitKey(1)
            if key == 27:
                break
            if mean > 200:
                logging.info("mean == 200 @ step %d" % step)
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    exit(main())