import pygame
import random
import sys
import math

class RaceCar:
    def __init__(self):
        pygame.init()
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)
        
        # Game constants
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 600
        self.FPS = 60
        
        # Car constants
        self.CAR_WIDTH = 40
        self.CAR_HEIGHT = 20
        self.CAR_SPEED = 5
        self.CAR_TURN_SPEED = 3
        self.CAR_START_X = self.WINDOW_WIDTH // 2
        self.CAR_START_Y = self.WINDOW_HEIGHT // 2
        
        # Map constants
        self.PIPE_THICKNESS = 20
        self.MIN_PIPE_LENGTH = 100
        self.MAX_PIPE_LENGTH = 200
        self.NUM_PIPES = 10
        
        # Game setup
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Race Car")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)
        self.resetGame()

    def resetGame(self):
        # Game variables
        self.carX = self.CAR_START_X
        self.carY = self.CAR_START_Y
        self.carAngle = 0
        self.pipes = []
        self.score = 0
        self.gameOver = False
        
        # Generate random pipes
        self.generatePipes()
        return self.getState()

    def generatePipes(self):
        self.pipes = []
        for _ in range(self.NUM_PIPES):
            # Random pipe position
            pipeX = random.randint(0, self.WINDOW_WIDTH - self.PIPE_THICKNESS)
            pipeY = random.randint(0, self.WINDOW_HEIGHT - self.PIPE_THICKNESS)
            
            # Random pipe length and orientation
            pipeLength = random.randint(self.MIN_PIPE_LENGTH, self.MAX_PIPE_LENGTH)
            isHorizontal = random.choice([True, False])
            
            if isHorizontal:
                pipe = pygame.Rect(pipeX, pipeY, pipeLength, self.PIPE_THICKNESS)
            else:
                pipe = pygame.Rect(pipeX, pipeY, self.PIPE_THICKNESS, pipeLength)
            
            self.pipes.append(pipe)

    def updateCar(self, action):
        # Convert angle to radians
        angleRad = math.radians(self.carAngle)
        
        # Calculate movement based on action
        if action == 0:  # Forward
            self.carX += math.cos(angleRad) * self.CAR_SPEED
            self.carY += math.sin(angleRad) * self.CAR_SPEED
        elif action == 1:  # Turn left
            self.carAngle += self.CAR_TURN_SPEED
            if self.carAngle >= 360:
                self.carAngle -= 360
        elif action == 2:  # Turn right
            self.carAngle -= self.CAR_TURN_SPEED
            if self.carAngle < 0:
                self.carAngle += 360

        # Create car rectangle for collision detection
        carRect = pygame.Rect(
            self.carX - self.CAR_WIDTH // 2,
            self.carY - self.CAR_HEIGHT // 2,
            self.CAR_WIDTH,
            self.CAR_HEIGHT
        )

        # Check collision with pipes
        for pipe in self.pipes:
            if carRect.colliderect(pipe):
                self.gameOver = True
                return -10  # Negative reward for collision

        # Check if car is out of bounds
        if (self.carX < 0 or self.carX > self.WINDOW_WIDTH or 
            self.carY < 0 or self.carY > self.WINDOW_HEIGHT):
            self.gameOver = True
            return -10  # Negative reward for going out of bounds

        return 0.1  # Small positive reward for staying alive

    def updateUI(self):
        self.screen.fill(self.WHITE)
        
        # Draw pipes
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, self.BLACK, pipe)
        
        # Draw car
        carSurface = pygame.Surface((self.CAR_WIDTH, self.CAR_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(carSurface, self.RED, (0, 0, self.CAR_WIDTH, self.CAR_HEIGHT))
        
        # Rotate car
        rotatedCar = pygame.transform.rotate(carSurface, -self.carAngle)
        carRect = rotatedCar.get_rect(center=(self.carX, self.carY))
        
        self.screen.blit(rotatedCar, carRect)
        
        # Draw score
        self.screen.blit(
            self.font.render(f"Score: {self.score}", True, self.BLACK),
            (10, 10)
        )
        
        pygame.display.flip()

    def getState(self):
        # Get distances to nearest pipes in 8 directions
        distances = []
        sensorAngles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        for angle in sensorAngles:
            distance = self.getDistanceToPipe(angle)
            distances.append(distance / max(self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        
        # Add car's current angle and position
        state = distances + [
            self.carAngle / 360,  # Normalized angle
            self.carX / self.WINDOW_WIDTH,  # Normalized X position
            self.carY / self.WINDOW_HEIGHT   # Normalized Y position
        ]
        
        return state

    def getDistanceToPipe(self, angle):
        # Convert angle to radians
        angleRad = math.radians(angle + self.carAngle)
        
        # Start from car position
        checkX, checkY = self.carX, self.carY
        
        # Check for pipe collision in the given direction
        while 0 <= checkX <= self.WINDOW_WIDTH and 0 <= checkY <= self.WINDOW_HEIGHT:
            # Create a small rectangle to check for collision
            checkRect = pygame.Rect(checkX - 1, checkY - 1, 2, 2)
            
            # Check collision with pipes
            for pipe in self.pipes:
                if checkRect.colliderect(pipe):
                    return math.sqrt((checkX - self.carX)**2 + (checkY - self.carY)**2)
            
            # Move in the direction of the angle
            checkX += math.cos(angleRad) * 5
            checkY += math.sin(angleRad) * 5
        
        return max(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)  # Return maximum distance if no pipe found

    def gameLoop(self, action):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Update game state
        reward = self.updateCar(action)
        
        # Update UI
        self.updateUI()
        self.clock.tick(self.FPS)
        
        # Update score
        self.score += 1
        
        return self.getState(), reward
        