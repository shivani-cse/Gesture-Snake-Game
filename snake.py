import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import sys
import random

# Configuration
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
SIDEBAR_WIDTH = 350
GAME_WIDTH = WINDOW_WIDTH - SIDEBAR_WIDTH

# Colors
COLOR_BG_GAME = (20, 20, 25)
COLOR_BG_SIDEBAR = (10, 10, 15)
COLOR_SNAKE_BODY = (46, 204, 113)
COLOR_SNAKE_OUTLINE = (30, 130, 76)
COLOR_SNAKE_HEAD = (241, 196, 15)
COLOR_FOOD = (231, 76, 60)
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (52, 152, 219)

# Physics
FPS = 60
SNAKE_SPEED = 14
SNAKE_RADIUS = 16
GROWTH_FACTOR = 4
START_LENGTH = 15

CAMERA_MARGIN = 0.25

# Smooths coordinate jitter
class Smoother:
    def __init__(self, alpha=0.15):
        self.x = 0.5
        self.y = 0.5
        self.alpha = alpha

    def update(self, tx, ty):
        self.x = self.x * (1 - self.alpha) + tx * self.alpha
        self.y = self.y * (1 - self.alpha) + ty * self.alpha
        return self.x, self.y

def map_range(value, in_min, in_max, out_min, out_max):
    norm = (value - in_min) / (in_max - in_min)
    norm = max(0.0, min(1.0, norm))
    return out_min + norm * (out_max - out_min)

# Vision System
class VisionSystem:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detected = False
        self.restart_gesture = False
        self.raw_finger_norm = (0.5, 0.5)
        self.game_finger_norm = (0.5, 0.5)
        self.smoother = Smoother()

    def update(self):
        ret, frame = self.cap.read()
        if not ret: return None

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        self.detected = False
        self.restart_gesture = False

        if results.multi_hand_landmarks:
            self.detected = True
            lm = results.multi_hand_landmarks[0]
            
            raw_x, raw_y = lm.landmark[8].x, lm.landmark[8].y
            self.raw_finger_norm = (raw_x, raw_y)

            mapped_x = map_range(raw_x, CAMERA_MARGIN, 1.0 - CAMERA_MARGIN, 0.0, 1.0)
            mapped_y = map_range(raw_y, CAMERA_MARGIN, 1.0 - CAMERA_MARGIN, 0.0, 1.0)
            
            self.game_finger_norm = self.smoother.update(mapped_x, mapped_y)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, lm, self.mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
            )

            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]
            up = [(lm.landmark[t].y < lm.landmark[p].y) for t, p in zip(tips, pips)]
            if up[0] and up[1] and not up[2] and not up[3]:
                self.restart_gesture = True

        margin_x = int(w * CAMERA_MARGIN)
        margin_y = int(h * CAMERA_MARGIN)
        cv2.rectangle(frame, (margin_x, margin_y), (w - margin_x, h - margin_y), (0, 255, 255), 2)

        return frame

    def release(self):
        self.cap.release()

# Game Engine
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Gesture Snake")
        self.clock = pygame.time.Clock()
        self.vision = VisionSystem()
        
        self.font_big = pygame.font.SysFont("arial rounded mt bold", 60)
        self.font_med = pygame.font.SysFont("arial rounded mt bold", 30)
        self.font_sml = pygame.font.SysFont("consolas", 18)
        self.font_ui_bold = pygame.font.SysFont("arial", 22, bold=True)

        self.cam_surface = None
        self.reset_game()
        self.state = "MENU"

    def reset_game(self):
        cx = SIDEBAR_WIDTH + (GAME_WIDTH // 2)
        cy = WINDOW_HEIGHT // 2
        
        self.head_pos = [float(cx), float(cy)]
        self.body = []
        
        for i in range(START_LENGTH):
            self.body.append([cx, cy + i*10])
            
        self.target_len = START_LENGTH
        self.score = 0
        self.spawn_food()
        self.start_ticks = 0
        self.invincible = True

    def spawn_food(self):
        pad = 50
        self.food_pos = [
            random.randint(SIDEBAR_WIDTH + pad, WINDOW_WIDTH - pad),
            random.randint(pad, WINDOW_HEIGHT - pad)
        ]

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: running = False

            frame = self.vision.update()
            if frame is not None:
                h, w, _ = frame.shape
                dim = min(h, w)
                cx, cy = w//2, h//2
                crop = frame[cy-dim//2:cy+dim//2, cx-dim//2:cx+dim//2]
                crop = cv2.resize(crop, (SIDEBAR_WIDTH - 40, SIDEBAR_WIDTH - 40))
                
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                self.cam_surface = pygame.surfarray.make_surface(np.transpose(crop_rgb, (1, 0, 2)))

                if self.vision.restart_gesture and self.state == "GAMEOVER":
                    self.reset_game()
                    self.state = "PLAYING"
                    self.start_ticks = pygame.time.get_ticks()

            if self.state == "MENU":
                if self.vision.detected:
                    self.state = "PLAYING"
                    self.start_ticks = pygame.time.get_ticks()

            elif self.state == "PLAYING":
                if not self.vision.detected:
                    self.state = "PAUSED"
                else:
                    self.update_physics()

            elif self.state == "PAUSED":
                if self.vision.detected:
                    self.state = "PLAYING"

            self.draw()
            self.clock.tick(FPS)

        self.vision.release()
        pygame.quit()
        sys.exit()

    def update_physics(self):
        norm_x, norm_y = self.vision.game_finger_norm
        
        target_x = SIDEBAR_WIDTH + (norm_x * GAME_WIDTH)
        target_y = norm_y * WINDOW_HEIGHT

        hx, hy = self.head_pos
        dx = target_x - hx
        dy = target_y - hy
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 10: 
            move_x = (dx / dist) * SNAKE_SPEED
            move_y = (dy / dist) * SNAKE_SPEED
            self.head_pos[0] += move_x
            self.head_pos[1] += move_y
            
            self.body.insert(0, list(self.head_pos))
            if len(self.body) > self.target_len:
                self.body.pop()

        if pygame.time.get_ticks() - self.start_ticks > 3000:
            self.invincible = False

        if not (SIDEBAR_WIDTH <= self.head_pos[0] <= WINDOW_WIDTH and 0 <= self.head_pos[1] <= WINDOW_HEIGHT):
            if not self.invincible: self.state = "GAMEOVER"
            else: 
                self.head_pos[0] = max(SIDEBAR_WIDTH, min(self.head_pos[0], WINDOW_WIDTH))
                self.head_pos[1] = max(0, min(self.head_pos[1], WINDOW_HEIGHT))

        if not self.invincible and len(self.body) > 20:
            for part in self.body[15:]:
                if math.hypot(self.head_pos[0]-part[0], self.head_pos[1]-part[1]) < SNAKE_RADIUS:
                    self.state = "GAMEOVER"

        fx, fy = self.food_pos
        if math.hypot(self.head_pos[0]-fx, self.head_pos[1]-fy) < (SNAKE_RADIUS + 15):
            self.score += 10
            self.target_len += GROWTH_FACTOR
            self.spawn_food()

    def draw(self):
        self.screen.fill(COLOR_BG_GAME)
        
        pygame.draw.rect(self.screen, COLOR_BG_SIDEBAR, (0, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(self.screen, (50, 50, 60), (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, WINDOW_HEIGHT), 2)

        for x in range(SIDEBAR_WIDTH, WINDOW_WIDTH, 80):
            pygame.draw.line(self.screen, (30, 30, 35), (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, 80):
            pygame.draw.line(self.screen, (30, 30, 35), (SIDEBAR_WIDTH, y), (WINDOW_WIDTH, y))

        fx, fy = int(self.food_pos[0]), int(self.food_pos[1])
        pygame.draw.circle(self.screen, COLOR_FOOD, (fx, fy), 14)
        pygame.draw.circle(self.screen, (255, 100, 100), (fx-4, fy-4), 5)
        pygame.draw.ellipse(self.screen, (46, 204, 113), (fx-2, fy-20, 8, 12))

        for i, pos in enumerate(reversed(self.body)):
            x, y = int(pos[0]), int(pos[1])
            color = COLOR_SNAKE_BODY if i % 2 == 0 else (40, 180, 100)
            pygame.draw.circle(self.screen, COLOR_SNAKE_OUTLINE, (x, y), SNAKE_RADIUS + 2)
            pygame.draw.circle(self.screen, color, (x, y), SNAKE_RADIUS)

        hx, hy = int(self.head_pos[0]), int(self.head_pos[1])
        pygame.draw.circle(self.screen, COLOR_SNAKE_OUTLINE, (hx, hy), SNAKE_RADIUS + 4)
        pygame.draw.circle(self.screen, COLOR_SNAKE_HEAD, (hx, hy), SNAKE_RADIUS + 2)
        
        eye_offset_x = 8 if (self.head_pos[0] > self.body[min(len(self.body)-1, 1)][0]) else -8
        pygame.draw.circle(self.screen, (0, 0, 0), (hx + eye_offset_x, hy - 5), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), (hx + eye_offset_x, hy + 5), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (hx + eye_offset_x + 1, hy - 6), 1)

        margin = 20
        cam_y = 20
        pygame.draw.rect(self.screen, (0, 0, 0), (margin, cam_y, SIDEBAR_WIDTH - 40, SIDEBAR_WIDTH - 40))
        if self.cam_surface:
            self.screen.blit(self.cam_surface, (margin, cam_y))
            box_label = self.font_sml.render("KEEP HAND IN BOX", True, (255, 255, 0))
            self.screen.blit(box_label, (margin + 10, cam_y + 10))

        border_color = (0, 255, 0) if self.state == "PLAYING" else (255, 50, 50)
        pygame.draw.rect(self.screen, border_color, (margin, cam_y, SIDEBAR_WIDTH - 40, SIDEBAR_WIDTH - 40), 3)

        text_y = cam_y + SIDEBAR_WIDTH - 10
        pygame.draw.rect(self.screen, (30, 30, 40), (margin, text_y, SIDEBAR_WIDTH - 40, 100), border_radius=15)
        
        lbl_score = self.font_med.render("SCORE", True, (150, 150, 150))
        val_score = self.font_big.render(str(self.score), True, COLOR_ACCENT)
        self.screen.blit(lbl_score, (margin + 20, text_y + 15))
        self.screen.blit(val_score, (margin + 20, text_y + 45))

        inst_y = WINDOW_HEIGHT - 150
        inst_lines = [
            "HOW TO PLAY:",
            "1. Move hand to guide Snake",
            "2. Keep hand in the yellow box",
            "3. Eat apples to grow",
            "4. Don't hit walls or yourself"
        ]
        
        for i, line in enumerate(inst_lines):
            c = (255, 255, 255) if i == 0 else (170, 170, 170)
            f = self.font_ui_bold if i == 0 else self.font_sml
            txt = f.render(line, True, c)
            self.screen.blit(txt, (margin, inst_y + i*25))

        cx = SIDEBAR_WIDTH + (GAME_WIDTH // 2)
        cy = WINDOW_HEIGHT // 2

        if self.state == "MENU":
            self.draw_overlay("READY?", "Raise hand to start", (0, 255, 0))
        elif self.state == "PAUSED":
            self.draw_overlay("PAUSED", "Lost Hand Tracking", (255, 200, 0))
        elif self.state == "GAMEOVER":
            self.draw_overlay("GAME OVER", "Show 'Peace Sign' to Restart", (255, 50, 50))
        elif self.state == "PLAYING" and self.invincible:
             txt = self.font_med.render("WARM UP...", True, (100, 255, 100))
             rect = txt.get_rect(center=(cx, cy - 200))
             self.screen.blit(txt, rect)

        pygame.display.flip()

    def draw_overlay(self, title, sub, color):
        cx = SIDEBAR_WIDTH + (GAME_WIDTH // 2)
        cy = WINDOW_HEIGHT // 2
        
        box_w, box_h = 400, 200
        s = pygame.Surface((box_w, box_h))
        s.set_alpha(220)
        s.fill((0,0,0))
        self.screen.blit(s, (cx - box_w//2, cy - box_h//2))
        
        pygame.draw.rect(self.screen, color, (cx - box_w//2, cy - box_h//2, box_w, box_h), 2)
        
        t1 = self.font_big.render(title, True, color)
        t2 = self.font_med.render(sub, True, (200, 200, 200))
        
        self.screen.blit(t1, t1.get_rect(center=(cx, cy - 30)))
        self.screen.blit(t2, t2.get_rect(center=(cx, cy + 30)))

if __name__ == "__main__":
    game = SnakeGame()
    game.run()
  
