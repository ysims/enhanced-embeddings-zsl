import os
import re
import pickle
import argparse
import torch

# Class/word embeddings (Word2Vec)
import gensim
from word2vec import word2vec

# Audio embedding inference
from AudioEmbeddings.models.YAMNet import YAMNet
from AudioEmbeddings.models.Inception import InceptionV4
from AudioEmbeddings.models.VGGish import VGGish
from AudioEmbeddings.inference import audio_to_embedding

# Command line arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument( "save_name", type=str, help="What name to give the saved data.")
parser.add_argument("model_path", type=str, help="Where the model is located.")
parser.add_argument("model", type=str, options=["YAMNet", "Inception", "VGGish"], help="Which model to use.")
parser.add_argument("--data_path", type=str, default="./ESC-50/audio/", help="Where the ESC-50 wav files are located.")
parser.add_argument("--device", type=str, default="auto", help="Device to run inference on.")
parser.add_argument("--synonyms", store_true=True, help="Use synonyms for the classes.")
args = parser.parse_args()
# fmt: on

# Check if cuda is available
args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

# Classes in ESC-50
# fmt: off
classes = ([
        "dog_canine_bark_woof_yap_call_animal_puppy", "rooster_cockerel_call_animal", "pig_hog_sow_swine_squeal_oink_grunt_call_animal", "cow_moo_call_bull_oxen_animal", "frog_toad_croak_call_animal", "cat_meow_mew_purr_hiss_chirp_kitten_feline_call_animal", "hen_cluck_chicken_animal_call", "insects_flying_buzz_hum_bug",  "sheep_bleat_animal_call_lamb", "crow_squawk_screech_caw_bird_call_cry_animal",
        "rain_drizzle_wet_sprinkle_shower_water_nature", "sea_waves_water_swell_tide_ocean_surf_nature", "crackling_fire_hissing_sizzling_flame_bonfire_campfire_nature", "crickets_insects_insect_bug_cicada_call", "chirping_birds_animal_call_song_tweet_chirp_twitter_trill_warble_chatter_cheep", "water_drops_splash_droplet_drip",  "wind_nature_gust_gale_blow_breeze_howl", "pouring_water_slosh_gargle_splash_splosh", "toilet_flush_water_flow_wash", "thunderstorm_thunder_storm_nature_lightning",
        "crying_baby_cry_human_whine_infant_child_wail_bawl_sob_scream_call", "sneezing_sneeze", "clapping_clap_applause_applaud_praise", "breathing_breath_breathe_gasp_exhale", "coughing_cough_hack", "footsteps_walking_walk_pace_step_gait_march", "laughing_cackle_laugh_chuckle_giggle_funny", "brushing_teeth_scrape_rub_brush", "snoring_snore_sleep_snore_snort_wheeze_breath", "drinking_sipping_gulp_gargle_drink_sip_breath",
        "door_wood_knock_tap_bang_thump", "mouse_click_computer_tap", "keyboard_typing_tap_mechanical_computer", "door_wood_creaks_squeak_creak_screech_scrape", "can_opening_hiss_fizz_air", "washing_machine_electrical_hum_thump_noise_loud", "vacuum_cleaner_electrical_noise_loud", "clock_alarm_signal_buzzer_alert_ring_beep", "clock_tick_tock_click_clack_beat_tap_ticking", "glass_breaking_crunch_crack_smash_clink_break_noise",
        "helicoper_chopping_engine_blades_whirring_swish_chopper_electrical_noise_vehicle_loud", "chainsaw_saw_electrical_noise_tool_loud", "siren_alarm_alert_bell_horn_noise_loud", "car_horn_vehicle_noise_blast_loud_honk", "engine_rumble_vehicle_chug_revving_car_drive", "train_clack_horn_clatter_vehicle_squeal_rattle", "church_bells_tintinnabulation_ring_chime_bell", "airplane_plane_motor_engine_hum_loud_noise", "fireworks_burst_bang_firecracker", "hand_saw_squeak_sawing_cut_hack_tool",
    ] if args.synonyms else [
        "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects_flying", "sheep", "crow", 
        "rain", "sea_waves", "crackling_fire",  "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
        "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
        "door_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
        "helicoper", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw",
    ])
# fmt: on

# Load the Word2Vec model
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
    "./word2vec.txt", binary=True
)

# Load the requested model
model = (
    YAMNet(channels=1, num_classes=30)
    if args.model == "YAMNet"
    else (
        InceptionV4(num_classes=30)
        if args.model == "Inception"
        else VGGish(num_classes=30)
    )
)
model.load_state_dict(torch.load(args.model_path, map_location=torch.device(args.device)))
model = model.to(args.device)
model.eval()

# The audio embedding function requires the number of channels
channels = 3 if args.model == "Inception" else 1

# Get all the files in the data path
files = os.listdir(args.data_path)

# Data is a dictionary of labels, auxiliary data (word2vec) and audio features (using the appropriate model for the seen/unseen split)
# Iterate on all of the wav files and create the dataset
data = {
    "labels": [int(re.split("-|\.", file)[3]) for file in files],
    "features": [audio_to_embedding(os.path.join(args.data_path, file), model, args.device, channels) for file in files],
    "auxiliary": [word2vec(w2v_model, classes[int(re.split("-|\.", file)[3])], double_first=True) for file in files],
}

# Make the directory to save the data if needed
folder_path = "./ESC_50_{}/".format("synonyms" if args.synonyms else "normal")
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# Save the data to a pickle file
filename = "./ESC_50_{}/{}.pickle".format(args.classes, args.save_name)
with open(filename, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(filename))
