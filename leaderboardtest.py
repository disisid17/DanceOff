import pygame
import csv
import os

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Leaderboard")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Fonts
font = pygame.font.Font(None, 36)

# Function to display leaderboard
def display_leaderboard():
    screen.fill(WHITE)
    title_text = font.render("Leaderboard", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 50))

    leaderboard = read_leaderboard()
    if leaderboard:
        y = 100
        for idx, entry in enumerate(leaderboard[:10]):
            text = font.render(f"{idx+1}. {entry[0]} - {entry[1]}", True, BLACK)
            screen.blit(text, (WIDTH // 2 - text.get_width() // 2, y))
            y += 40
    else:
        no_data_text = font.render("No data available", True, BLACK)
        screen.blit(no_data_text, (WIDTH // 2 - no_data_text.get_width() // 2, 100))

    pygame.display.flip()

# Function to read leaderboard from CSV file
def read_leaderboard():
    if not os.path.exists("leaderboard.csv"):
        return None
    with open("leaderboard.csv", "r") as file:
        reader = csv.reader(file)
        leaderboard = list(reader)
    leaderboard.sort(key=lambda x: int(x[1]), reverse=True)  # Sort by score
    return leaderboard

# Function to write score to leaderboard CSV
def write_to_leaderboard(name, score):
    with open("leaderboard.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([name, score])

# Main function
def nameToBoard(score):
    running = True
    name = ""
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RETURN:
                    write_to_leaderboard(name, score)
                    running = False
                elif event.key == pygame.K_BACKSPACE:
                    name = name[:-1]
                else:
                    name += event.unicode

        screen.fill(WHITE)
        input_text = font.render("Enter your name: " + name, True, BLACK)
        screen.blit(input_text, (WIDTH // 2 - input_text.get_width() // 2, HEIGHT // 2))

        pygame.display.flip()
        clock.tick(30)

    

# Example usage:
score = 5000  # Replace with your game's score
nameToBoard(score)
display_leaderboard()
# Wait for a while before closing
pygame.time.delay(5000)
pygame.quit()