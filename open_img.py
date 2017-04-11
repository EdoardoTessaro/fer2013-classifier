from read_dataset import get_dataset
import pygame

emotion = [
  "Anger",
  "Disgust",
  "Fear",
  "Happiness",
  "Sadness",
  "Surprise",
  "Neutral"
  ]
def draw_img(s, pic, label, x, y, font, draw_labels):
  text = emotion[label]
  for c in range(48):
    for r in range(48):
      pix = int(pic[c + r*48])
      color = pygame.Color(pix,pix,pix, 255)
      s.set_at((48*x + c, 48*y + r), color)
  if draw_labels:
    text_srf = font.render(text, 1, (255,255,255)) 
    s.blit(text_srf, (48*x, 48*y))
    s.blit(text_srf, (48*x+2, 48*y))
    s.blit(text_srf, (48*x, 48*y+2))
    s.blit(text_srf, (48*x+2, 48*y+2))
    text_srf = font.render(text, 1, (0,0,0))
    s.blit(text_srf, (48*x+1, 48*y+1))
  
  

num_img = 10
width = height = 48*num_img

print("Reading dataset...")
data, label, _, _ = get_dataset(0, num_img**2)
print("Done.")

pygame.init()

s = pygame.Surface((width, height))

font = pygame.font.SysFont("Arial", 8)
for y in range(num_img):
  for x in range(num_img):
    draw_img(s, data[x + y*num_img], label[x + y*num_img], x, y, font, draw_labels=False)

screen = pygame.display.set_mode((width, height))

screen.blit(s, (0,0))
pygame.display.flip()
while True:
  pass
