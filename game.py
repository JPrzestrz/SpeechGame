# Arcade library
import arcade 
import random
# Libraries for speech recognition 
import wave
import pyaudio
import numpy as np
import pathlib
import os
# Turing off the tensorflow warnings about GPU 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from func import decode_audio, get_label, get_waveform_and_label, get_spectrogram, plot_spectrogram, get_spectrogram_and_label_id, preprocess_dataset

# Screen
SCREEN_HEIGHT = 700
SCREEN_WIDTH = 600
# Coin 
COIN_COUNT = 2
COIN_SCALE = 0.075
# Start Menu 
BUTTON_SCALE = 0.5
# Character, directions, constants for storing facing 
CHARACTER_SCALE = 1.25
RIGHT_DIR = 1
LEFT_DIR = 0
# How fast to move, and how fast to run the animation
MOVEMENT_SPEED = 5
UPDATES_PER_FRAME = 5
# Obstacle const 
OBST_SCALE = 0.2
OBST_COUNT = 2
OBST_SPEED = 2
# Speech constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "wav_out/output.wav"
# Deep learning and testing vars
DATASET_PATH = 'data/mini_speech_commands'
data_dir = pathlib.Path(DATASET_PATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']
AUTOTUNE = tf.data.AUTOTUNE
MY_TEST_PATH = 'wav_out/'
data_dir = pathlib.Path(MY_TEST_PATH)
model = tf.keras.models.load_model('saved_model/my_model')

title = 'GameTest'
exit = False

def load_texture_pair(filename):
    """
    Load a texture pair, with the second being a mirror image for direction distinguishing.
    """
    return [
        arcade.load_texture(filename),
        arcade.load_texture(filename, flipped_horizontally=True)
    ]

class PlayerCharacter(arcade.Sprite):
    def __init__(self):

        # Set up parent class
        super().__init__()

        # Default to face-right
        self.character_face_direction = RIGHT_DIR

        # Used for flipping between image sequences
        self.cur_texture = 0
        self.scale = CHARACTER_SCALE

        # Adjust the collision box. Default includes too much empty space
        # side-to-side. Box is centered at sprite center, (0, 0)
        # self.points = [[-22, -64], [22, -64], [22, 28], [-22, 28]]

        # --- Load Textures ---
        # Image path
        player_path = "images/player"

        # Load textures for idle standing
        self.idle_texture_pair = load_texture_pair(f"{player_path}0.png")

        # Load textures for walking
        self.walk_textures = []
        for i in range(6):
            texture = load_texture_pair(f"{player_path}{i}.png")
            self.walk_textures.append(texture)

    def update_animation(self, delta_time: float = 1 / 60):

        # Figure out if we need to flip face left or right
        if self.change_x < 0 and self.character_face_direction == RIGHT_DIR:
            self.character_face_direction = LEFT_DIR
        elif self.change_x > 0 and self.character_face_direction == LEFT_DIR:
            self.character_face_direction = RIGHT_DIR

        # Idle animation
        if self.change_x == 0 and self.change_y == 0:
            self.texture = self.idle_texture_pair[self.character_face_direction]
            return

        # Walking animation
        self.cur_texture += 1
        if self.cur_texture > 4 * UPDATES_PER_FRAME:
            self.cur_texture = 0
        frame = self.cur_texture // UPDATES_PER_FRAME
        direction = self.character_face_direction
        self.texture = self.walk_textures[frame+1][direction]

class MyGameWindow(arcade.Window):
    def __init__(self,width,height,title):
        super().__init__(SCREEN_WIDTH,SCREEN_HEIGHT,title) 
        self.set_location(400,50)
        arcade.set_background_color(arcade.color.RED_DEVIL)

    def setup(self):
        '''
        Initial settings of game window
        '''
        # Generate boxes 
        self.background = arcade.load_texture('images/menu_bg.png')
        # Starting Menu Graphics
        self.start_button = arcade.Sprite("images/start_button.png", scale=COIN_SCALE)
        self.start_button.center_x = SCREEN_WIDTH // 2
        self.start_button.center_y = SCREEN_HEIGHT // 2
        self.start_button.scale = BUTTON_SCALE
        self.floor = arcade.Sprite("images/bottom_bar.png", scale=1,center_x=SCREEN_WIDTH//2,center_y=10) 
        self.steer_logo = arcade.Sprite("images/speech.png", scale=0.5)
        self.steer_logo.center_x = SCREEN_WIDTH - self.steer_logo.width//2
        self.steer_logo.center_y = self.steer_logo.height//2
        # Lists setup
        self.player_list = arcade.SpriteList()
        self.obst_list = arcade.SpriteList()
        self.coin_list = arcade.SpriteList()
        # Player setup 
        self.player = PlayerCharacter()
        self.player.center_x = SCREEN_WIDTH // 2
        self.player.center_y = 100
        self.player.scale = CHARACTER_SCALE
        self.player_list.append(self.player)
        # Game values 
        # Steering: 0 - keyboard, 1 - voice 
        self.score = 0
        self.timer = 0
        self.steering = 1
        self.delta = 0
        # dividing game into scenes to make it more attractive
        # there will be main menu - 0, game - 1, pause - 2
        self.scene = 0
        # Speech variables 
        self.speech_active = 0
        self.speech_dir = 0 
        self.speech_change = 0
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK)
        self.frames = []
        # obstacles spawning 
        for i in range(OBST_COUNT):
            obst = arcade.Sprite('images/obstacle.png', scale=OBST_SCALE)
            obst.center_x = random.randrange(50,SCREEN_WIDTH-50)
            obst.center_y = i*250 + SCREEN_HEIGHT
            self.obst_list.append(obst)

        # coins spawning
        for i in range(COIN_COUNT):
            coin = arcade.Sprite("images/coin.png", scale=COIN_SCALE)
            # Spawning coins in distance of 100px from player 
            coin.center_x = random.randrange(SCREEN_WIDTH)
            if coin.center_x - self.player.center_x < 100 and coin.center_x - self.player.center_x > 0:
                coin.center_x += 100
            elif coin.center_x - self.player.center_x > -100 and coin.center_x - self.player.center_x < 0:
                coin.center_x -= 100
            coin.center_y = 100
            coin.center_x = coin.center_x % SCREEN_WIDTH
            self.coin_list.append(coin)

    def on_draw(self):
        """
        Render the screen.
        """
        # This command has to happen before we start drawing
        self.clear()
        # Draw the background texture
        arcade.draw_lrwh_rectangle_textured(0, 0,SCREEN_WIDTH, SCREEN_HEIGHT,self.background)
        if self.scene == 0:
            # Draw menu etc
            #print("Here is menu")
            self.start_button.draw()

        if self.scene == 1 or self.scene == 2:
            # Draw all the sprites.
            self.floor.draw()
            self.steer_logo.draw()
            self.player_list.draw()
            self.coin_list.draw()
            self.obst_list.draw()

            # Draw score on the screen
            score_text = f"Score: {self.score}"
            arcade.draw_text(score_text,
                            start_x=10,
                            start_y=10,
                            color=arcade.csscolor.WHITE,
                            font_size=18)
            if self.scene == 2:
                arcade.Sprite('images/pause.png', scale = 1, center_x = SCREEN_WIDTH//2, center_y = SCREEN_HEIGHT//2).draw()
        elif self.scene == 3:
            arcade.Sprite('images/pause.png', scale = 1, center_x = SCREEN_WIDTH//2, center_y = SCREEN_HEIGHT//2).draw()
            final_score = "FINAL SCORE"
            arcade.draw_text(final_score,
                            start_x=0,
                            start_y=SCREEN_HEIGHT//2,
                            color=arcade.csscolor.WHITE,
                            font_size=24,
                            width = SCREEN_WIDTH,
                            align = 'center')
            arcade.draw_text(self.score,
                            start_x=0,
                            start_y=SCREEN_HEIGHT//2 - 30,
                            color=arcade.csscolor.WHITE,
                            font_size=24,
                            width = SCREEN_WIDTH,
                            align = 'center')
            

    def on_key_press(self, key, modifiers):
        """
        Called whenever a key is pressed.
        """
        if key == arcade.key.R and self.scene == 3:
            self.setup()
        if self.scene == 1 or self.scene == 2:
            if key == arcade.key.LEFT and self.steering == 0:
                self.player.change_x = -MOVEMENT_SPEED
            if key == arcade.key.RIGHT and self.steering == 0:
                self.player.change_x = MOVEMENT_SPEED
            if key == arcade.key.S:
                if self.steering == 0:
                    self.steering = 1
                    self.steer_logo = arcade.Sprite("images/speech.png", scale=0.5)
                else:
                    self.steering = 0
                    self.steer_logo = arcade.Sprite("images/not_speech.png", scale=0.5)
                self.steer_logo.center_x = SCREEN_WIDTH - self.steer_logo.width//2
                self.steer_logo.center_y = self.steer_logo.height//2
            if key == arcade.key.SPACE and self.steering == 1:
                # Getting 1 sec of speech from microphone 
                self.speech_active = 1
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK)
                self.frames = []

    def on_key_release(self, key, modifiers):
        """
        Called when the user releases a key.
        """
        if self.scene == 1 or self.scene == 2:
            if (key == arcade.key.LEFT or key == arcade.key.RIGHT) and self.steering == 0:
                self.player.change_x = 0
            if key == arcade.key.SPACE and self.steering == 1:
                # Getting 1 sec of speech from microphone 
                self.speech_active = 2

    def on_mouse_press(self, x, y, button, modifiers):
        """
        Called when the left mouse button is pressed.
        """
        vert = (x < self.start_button.center_x + self.start_button.width // 2 and  x > self.start_button.center_x - self.start_button.width // 2)
        hor = (y < self.start_button.center_y + self.start_button.height // 2 and  y > self.start_button.center_y - self.start_button.height // 2)
        if vert and hor and self.scene==0:
            self.scene = 1 
            self.background = arcade.load_texture('images/game_bg.png')

    def on_update(self, delta_time):
        """ Movement and game logic """
        # Move the obstacles
        if self.scene == 1:
            for obst in self.obst_list:
                obst.center_y -= OBST_SPEED
        
        # Delta for better performance while reading mic         
        self.delta += 1 
        if self.delta >= 4:
            self.delta = 0

        # Adding frame of recording from microphone 
        # and appending it to the list     
        if (self.speech_active == 1 and self.delta == 0) or (self.speech_active == 2 and self.delta == 4) :
            # Collect voice in "push to talk" manner
            data = self.stream.read(CHUNK)
            self.frames.append(data)

        if self.speech_change > 20:
            self.speech_change = 0
            self.speech_dir = 0 
            self.player.change_x = 0

        if self.speech_dir == 2:
                self.player.change_x = -MOVEMENT_SPEED
                self.speech_change += 1
        if self.speech_dir == 1:
            self.player.change_x = MOVEMENT_SPEED
            self.speech_change += 1

        # When the SPACEBAR is released we collect 
        # all the frames and save the recording 
        if self.speech_active == 2:
            #print("*done recording")
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            self.frames=[]
            self.speech_active = 3

        if self.speech_active == 3:
            filenames = tf.io.gfile.glob(str(data_dir) + '/*')
            #print('\n\nNumber of total examples:', len(filenames))
            # Processing Files
            test_ds = preprocess_dataset(filenames)
            # Dividing into audio and labels 
            # no or wrong labels in mydata files
            test_audio = []
            test_labels = []
            for audio, label in test_ds:
                test_audio.append(audio.numpy())
                test_labels.append(label.numpy())
            test_audio = np.array(test_audio)
            test_labels = np.array(test_labels)
            # Using our trained model to predict audio files
            # included in mydata dir
            y_pred = np.argmax(model.predict(test_audio), axis=1)
            for i in y_pred:
                #print(f'Command: {commands[i]}, label: {i}')
                if i == 4:
                    # Go right 
                    self.speech_dir = 1
                elif i == 2:
                    # Go left 
                    self.speech_dir = 2
                elif i == 5:
                    # Pause the game 
                    self.scene = 2
                elif i == 1:
                    # Rseume the game
                    if self.scene == 2:
                        self.scene = 1
                    elif self.scene == 3:
                        self.setup()
            self.speech_active = 0

        # Move the player
        if self.player.center_x < 25:
            self.player.change_x = 0 
            self.player.center_x +=MOVEMENT_SPEED
        if self.player.center_x > SCREEN_WIDTH - 25:
            self.player.change_x = 0 
            self.player.center_x -=MOVEMENT_SPEED
        self.player_list.update()
        # Visual addition - moving start button 
        self.timer +=1
        if self.timer >= 12 and self.start_button.scale==BUTTON_SCALE:
            self.start_button.scale+=0.01
            self.timer=0
        elif self.timer >= 12 and self.start_button.scale!=BUTTON_SCALE:
            self.start_button.scale-=0.01
            self.timer=0
        # Update the players animation
        self.player_list.update_animation()

        # Generate a list of all sprites that collided with the player.
        hit_coins = arcade.check_for_collision_with_list(self.player, self.coin_list)
        hit_obst = arcade.check_for_collision_with_list(self.player,self.obst_list)
        hit_floor = arcade.check_for_collision_with_list(self.floor,self.obst_list)
        
        # Loop through each colliding sprite, remove it, and add to the score.
        for coin in hit_coins:
            coin.remove_from_sprite_lists()
            self.score += 100
            # Adding coin after hitting one 
            coin = arcade.Sprite("images/coin.png", scale=COIN_SCALE)
            coin.center_x = random.randrange(SCREEN_WIDTH)
            if coin.center_x - self.player.center_x < 100 and coin.center_x - self.player.center_x > 0:
                coin.center_x += 100
            elif coin.center_x - self.player.center_x > -100 and coin.center_x - self.player.center_x < 0:
                coin.center_x -= 100
            coin.center_x = coin.center_x % SCREEN_WIDTH
            coin.center_y = 100
            self.coin_list.append(coin)

        for obst in hit_floor:
            # Destroying obstacles that touched the floor 
            obst.remove_from_sprite_lists()
            # Generating new obstacles 
            obst = arcade.Sprite('images/obstacle.png', scale=OBST_SCALE)
            obst.center_x = random.randrange(50,SCREEN_WIDTH-50)
            obst.center_y = random.randrange(0,300) + SCREEN_HEIGHT
            self.obst_list.append(obst)

        for obst in hit_obst:
            obst.remove_from_sprite_lists()
            self.scene = 3
        
window = MyGameWindow(SCREEN_WIDTH,SCREEN_HEIGHT,title)
# Window setup 
window.setup()
arcade.run()