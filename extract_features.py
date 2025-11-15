import numpy as np
import pandas as pd
from tqdm import tqdm

# Mapping of move types relative to Pokémon types, highlighting weaknesses and strengths
TYPE_CHART = {
    'NORMAL': {
        'ROCK': 0.5, 'GHOST': 0, 'STEEL': 0.5
    },
    'FIRE': {
        'FIRE': 0.5, 'WATER': 0.5, 'GRASS': 2, 'ICE': 2, 'BUG': 2, 'ROCK': 0.5, 'DRAGON': 0.5, 'STEEL': 2
    },
    'WATER': {
        'FIRE': 2, 'WATER': 0.5, 'GRASS': 0.5, 'GROUND': 2, 'ROCK': 2, 'DRAGON': 0.5
    },
    'GRASS': {
        'FIRE': 0.5, 'WATER': 2, 'GRASS': 0.5, 'POISON': 0.5, 'GROUND': 2, 'FLYING': 0.5, 'BUG': 0.5, 'ROCK': 2, 'DRAGON': 0.5, 'STEEL': 0.5
    },
    'ELECTRIC': {
        'WATER': 2, 'GRASS': 0.5, 'ELECTRIC': 0.5, 'GROUND': 0, 'FLYING': 2, 'DRAGON': 0.5
    },
    'ICE': {
        'FIRE': 0.5, 'WATER': 0.5, 'GRASS': 2, 'ICE': 0.5, 'GROUND': 2, 'FLYING': 2, 'DRAGON': 2, 'STEEL': 0.5
    },
    'FIGHTING': {
        'NORMAL': 2, 'ICE': 2, 'POISON': 0.5, 'FLYING': 0.5, 'PSYCHIC': 0.5, 'BUG': 0.5, 'ROCK': 2, 'GHOST': 0, 'DARK': 2, 'STEEL': 2, 'FAIRY': 0.5
    },
    'POISON': {
        'GRASS': 2, 'POISON': 0.5, 'GROUND': 0.5, 'ROCK': 0.5, 'GHOST': 0.5, 'STEEL': 0, 'FAIRY': 2
    },
    'GROUND': {
        'FIRE': 2, 'ELECTRIC': 2, 'GRASS': 0.5, 'POISON': 2, 'FLYING': 0, 'BUG': 0.5, 'ROCK': 2, 'STEEL': 2
    },
    'FLYING': {
        'GRASS': 2, 'ELECTRIC': 0.5, 'FIGHTING': 2, 'BUG': 2, 'ROCK': 0.5, 'STEEL': 0.5
    },
    'PSYCHIC': {
        'FIGHTING': 2, 'POISON': 2, 'PSYCHIC': 0.5, 'DARK': 0, 'STEEL': 0.5
    },
    'BUG': {
        'FIRE': 0.5, 'GRASS': 2, 'FIGHTING': 0.5, 'POISON': 0.5, 'FLYING': 0.5, 'PSYCHIC': 2, 'GHOST': 0.5, 'DARK': 2, 'STEEL': 0.5, 'FAIRY': 0.5
    },
    'ROCK': {
        'FIRE': 2, 'ICE': 2, 'FIGHTING': 0.5, 'GROUND': 0.5, 'FLYING': 2, 'BUG': 2, 'STEEL': 0.5
    },
    'GHOST': {
        'NORMAL': 0, 'PSYCHIC': 2, 'GHOST': 2, 'DARK': 0.5
    },
    'DRAGON': {
        'DRAGON': 2, 'STEEL': 0.5, 'FAIRY': 0
    },
    'DARK': {
        'FIGHTING': 0.5, 'PSYCHIC': 2, 'GHOST': 2, 'DARK': 0.5, 'FAIRY': 0.5
    },
    'STEEL': {
        'FIRE': 0.5, 'WATER': 0.5, 'ELECTRIC': 0.5, 'ICE': 2, 'ROCK': 2, 'STEEL': 0.5, 'FAIRY': 2
    },
    'FAIRY': {
        'FIRE': 0.5, 'FIGHTING': 2, 'POISON': 0.5, 'DRAGON': 2, 'DARK': 2, 'STEEL': 0.5
    }
    }

POSITIVE_EFFECTS = {
    "reflect",          # halves physical damage
    "lightscreen",      # halves special damage
    "auroraveil",       # halves both types of damage (if it's hail)
    "substitute",       # creates a temporary shield
    "tailwind",         # doubles Speed
    "focusenergy",      # increases critical-hit chance
    "safeguard",        # prevents status conditions
    "aqua_ring",        # recovers HP each turn
    "ingrain",          # recovers HP each turn but prevents switching
    "leechseed" ,       # (only if used on the opponent) drains HP each turn
    "wish",             # heals in a future turn
    "protect",          # blocks damage for one turn
    "detect",           # like protect
    "spikes",           # places hazards on the opponent's field
    "toxicspikes",      # poisons Pokémon that switch in
    "stealthrock",      # damages Pokémon that switch in
    "stickyweb",        # reduces Speed of the Pokémon that switches in
    "trickroom",        # reverses speed priority
    "psychicterrain", "mistyterrain", "grassyterrain", "electricterrain",  # beneficial terrains
    "rain", "sun", "sandstorm", "hail",  # weather, beneficial for certain teams
}

NEGATIVE_EFFECTS = {
    "leechseed",        # if suffered (drains each turn)
    "confusion",        # may damage itself
    "curse",            # takes damage each turn (if from Ghost)
    "bind", "wrap", "fire_spin", "clamp", "whirlpool",  # trapped and takes damage
    "infestation", "sand_tomb", "magma_storm",          # modern versions of binding
    "encore",           # forced to repeat a move
    "taunt",            # can only use offensive moves
    "torment",          # cannot use the same move consecutively
    "infatuation",      # may not attack
    "disable",          # disables a move
    "yawn",             # will fall asleep on the next turn
    "perishsong",       # KO after 3 turns
    "embargo",          # cannot use items
    "healblock",        # cannot heal
    "nightmare",        # takes damage while sleeping
    "partiallytrapped", # generic: trapped
}

def calculate_moves_effectiveness(move_type: str, defender_types: list) -> float:
    """
    Calculates the effectiveness of a move against a Pokémon.
    """
    if not move_type or not defender_types:
        return 1.0
    
    effectiveness = 1.0
    
    # For each defender type, multiply the effectiveness
    for def_type in defender_types:
        if move_type in TYPE_CHART:
            multiplier = TYPE_CHART[move_type].get(def_type, 1.0)
            effectiveness *= multiplier
    return effectiveness

def calculate_effectiveness(effects, positive, negative):
    for e in effects:
        if e in POSITIVE_EFFECTS:
            positive += 1
        elif e in NEGATIVE_EFFECTS:
            negative += 1
    return positive, negative
        
def extract_pokemon_p1(p1_team):
    pokemon_types = {}
    
    for pokemon in p1_team:
        name = pokemon.get('name', '').lower()
        types = [t.upper() for t in pokemon.get('types', [])]
        
        # Remove "NOTYPE"
        types = [t for t in types if t != 'NOTYPE']
        if name and types:
            pokemon_types[name] = types
    return pokemon_types

def moves_analyze(number: str, p_move, p_state, p_other_state, p_state_before, p_name_pok, pokemon_types, variables: dict) -> dict:
    move_type = p_move.get('type', '').upper()
    base_p = p_move.get('base_power', 0)
    category = p_move.get('category', '').upper()
    
    variables[f"p{number}_base_powers"].append(base_p)
    
    if category == 'SPECIAL':
        variables[f"p{number}_special_moves"] += 1
    elif category == 'PHYSICAL':
        variables[f"p{number}_physical_moves"] += 1

    
    if p_state_before and category in ['SPECIAL', 'PHYSICAL'] and base_p > 0:
        # Calculate observed damage
        hp_before = p_state_before.get('hp_pct', 1.0)
        hp_after = p_other_state.get('hp_pct', 1.0)
        
        # Check if it's the same Pokémon
        if p_state_before.get('name') == p_name_pok:
            damage_dealt = hp_before - hp_after
            
            # Only if there was positive damage
            if damage_dealt > 0:
                variables[f"p{number}_damages"].append(damage_dealt)
            elif damage_dealt == 0:
                variables[f"p{number}_no_attack"] += 1
            else:
                variables[f"p{1 if number == 2 else 2}_heal"] += 1
        else:
            variables[f"p{number}_num_change"] += 1

    if number == 2:
        if category != 'STATUS' and base_p > 0:
            defender_name = p_state.get('name', '').lower()
            defender_types = pokemon_types.get(defender_name, [])
            
            if defender_types:
                effectiveness = calculate_moves_effectiveness(move_type, defender_types)
            if effectiveness >= 2.0:
                variables[f"p{number}_super_effective"] += 1
            else:
                variables[f"p{number}_no_effective"] += 1
    return variables

def analyze_timeline(timeline, pokemon_types, variables: dict) -> dict:

    for i in range(len(timeline)):
        turn = timeline[i]

        # Move details for turn i of p1
        p1_move = turn.get('p1_move_details')
        p1_state = turn.get('p1_pokemon_state', {})
        p1_boosts = p1_state.get('boosts', {})
        p1_name_pok = p1_state.get("name")
        p1_state_before = timeline[i-1].get('p1_pokemon_state', {}) if i > 0 else None
        effects_p1 = p1_state.get('effects', [])
        
        variables["p1_list_hp"].append(p1_state.get('hp_pct', 0))

        # Move details for turn i of p2
        p2_move = turn.get('p2_move_details')
        p2_state = turn.get('p2_pokemon_state', {})
        p2_boosts = p2_state.get('boosts', {})
        p2_name_pok = p2_state.get("name")
        p2_state_before = timeline[i-1].get('p2_pokemon_state', {}) if i > 0 else None
        effects_p2 = p2_state.get('effects', [])
        
        variables["p2_list_hp"].append(p2_state.get('hp_pct', 0))

        if p1_move:
            variables = moves_analyze(1, p1_move, p1_state, p2_state, p2_state_before, p2_name_pok, pokemon_types, variables)

        if p1_boosts:
            total_boost = sum(p1_boosts.values())

            variables["p1_atk_boosts"].append(p1_boosts.get('atk', 0))
            variables["p1_def_boosts"].append(p1_boosts.get('def', 0))
            variables["p1_spa_boosts"].append(p1_boosts.get('spa', 0))
            variables["p1_spd_boosts"].append(p1_boosts.get('spd', 0))
            variables["p1_spe_boosts"].append(p1_boosts.get('spe', 0))
            
            if total_boost < 0:
                variables["p1_negative_boost_turns"] += 1

        if p2_move:
            variables = moves_analyze(2, p2_move, p2_state, p1_state, p1_state_before, p1_name_pok, pokemon_types, variables)

        if p2_boosts:
            # Count turns with total positive/negative boosts
            total_boost = sum(p2_boosts.values())

            variables["p2_atk_boosts"].append(p2_boosts.get('atk', 0))
            variables["p2_def_boosts"].append(p2_boosts.get('def', 0))
            variables["p2_spa_boosts"].append(p2_boosts.get('spa', 0))
            variables["p2_spd_boosts"].append(p2_boosts.get('spd', 0))
            variables["p2_spe_boosts"].append(p2_boosts.get('spe', 0))
            
            if total_boost < 0:
                variables["p2_negative_boost_turns"] += 1

        # Calculation of combined features between p1 and p2

        variables["rem_pok_p1"][p1_name_pok] = p1_state.get('hp_pct', 1.0)
        variables["rem_pok_p2"][p2_name_pok] = p2_state.get('hp_pct', 1.0)

        if p1_name_pok not in variables["pokemon_p1"]:
            variables["pokemon_p1"].append(p1_name_pok)

        if p2_name_pok not in variables["pokemon_p2"]:
            variables["pokemon_p2"].append(p2_name_pok)

        if p1_state.get("status", "nostatus") == "fnt":
            del variables["rem_pok_p1"][p1_name_pok]

        if p2_state.get("status", "nostatus") == "fnt":
            del variables["rem_pok_p2"][p2_name_pok]

        variables["p1_number_effective_pos"], variables["p1_number_effective_neg"] = calculate_effectiveness(effects_p1, variables["p1_number_effective_pos"], variables["p1_number_effective_neg"])
        variables["p2_number_effective_pos"], variables["p2_number_effective_neg"] = calculate_effectiveness(effects_p2, variables["p2_number_effective_pos"], variables["p2_number_effective_neg"])

    return variables

def create_simple_features(data: list[dict]) -> pd.DataFrame:
    """
    Feature extraction function based on correlation analysis with player victory.
    """
    feature_list = []

    for battle in tqdm(data, desc="Extracting features"):
        features = {}

        # --- P1 Team ---
        p1_team = battle.get('p1_team_details', [])
        team_size = len(p1_team)
        pokemon_types = extract_pokemon_p1(p1_team)

        # --- Battle Timeline Features ---
        timeline = battle.get('battle_timeline', [])

        variables = {
            # List of base_power values during the battle
            "p1_base_powers": [],
            "p2_base_powers": [],
            # Counters for special and physical moves for p1 and p2
            "p1_special_moves": 0,
            "p1_physical_moves": 0,
            "p2_special_moves": 0,
            "p2_physical_moves": 0,
            # Counters for total negative boosts
            "p1_negative_boost_turns": 0,
            "p2_negative_boost_turns": 0,
            # Counter for p2's super-effective moves against p1
            "p2_super_effective":0,
            # Counter for p2's non super-effective moves against p1
            "p2_no_effective": 0,
            # List of damages dealt
            "p1_damages": [],
            "p2_damages": [],
            # List of HP throughout the match
            "p1_list_hp": [],
            "p2_list_hp": [],
            # Counter for positive effects
            "p1_number_effective_pos": 0,
            "p2_number_effective_pos": 0,
            # Counter for negative effects
            "p1_number_effective_neg": 0,
            "p2_number_effective_neg": 0,
            # Lists of boosts per stat for P1
            "p1_atk_boosts": [],
            "p1_def_boosts": [],
            "p1_spa_boosts": [],
            "p1_spd_boosts": [],
            "p1_spe_boosts": [],
            # Lists of boosts per stat for P2
            "p2_atk_boosts": [],
            "p2_def_boosts": [],
            "p2_spa_boosts": [],
            "p2_spd_boosts": [],
            "p2_spe_boosts": [],
            # Counter for switches during the battle
            "p1_num_change": 0,
            "p2_num_change": 0,
            # List of Pokémon that appeared during the match
            "pokemon_p1": [],
            "pokemon_p2": [],
            # Counter for heals during the battle
            "p1_heal": 0,
            "p2_heal": 0,
            # Pokémon remaining at the 30th turn
            "rem_pok_p1": {},
            "rem_pok_p2": {},
            # Counter of attacks that failed
            "p1_no_attack": 0,
            "p2_no_attack": 0,
        }

        
        if timeline:

            variables = analyze_timeline(timeline, pokemon_types, variables)
            features['damage_advantage'] = sum(variables["p1_damages"]) - sum(variables["p2_damages"])
            features['var_pok'] = len(variables["pokemon_p1"]) - len(variables["pokemon_p2"])
            
            # Number of turns where p1's Pokémon are fnt (knocked out)
            ko_p1 = sum(
                1 for turn in timeline 
                if turn.get('p1_pokemon_state', {}).get('status', 'nostatus') == 'fnt'
            )

            # Number of turns where p2's Pokémon are fnt (knocked out)
            ko_p2 = sum(
                1 for turn in timeline 
                if turn.get('p2_pokemon_state', {}).get('status', 'nostatus') == 'fnt' 
            )
            
            features["ko_adv"] = ko_p1 - ko_p2

            # Assigning features
            features['negative_adv'] = variables["p1_negative_boost_turns"] - variables["p2_negative_boost_turns"]

            p1_rem_hp = np.mean([hp for hp in variables["rem_pok_p1"].values()]) if variables["rem_pok_p1"].values() else 0
            p2_rem_hp = np.mean([hp for hp in variables["rem_pok_p2"].values()]) if variables["rem_pok_p2"].values() else 0
            features["rem_hp_adv"] = p1_rem_hp - p2_rem_hp

            features['number_change'] = variables["p1_num_change"] - variables["p2_num_change"]

            features['heal_adv'] = variables["p1_heal"] - variables["p2_heal"]

            p1_effect_score = variables["p1_number_effective_pos"] - variables["p1_number_effective_neg"]
            p2_effect_score = variables["p2_number_effective_pos"] - variables["p2_number_effective_neg"]
            features['score_adv'] = p1_effect_score - p2_effect_score

            features['no_atk_adv'] = variables["p2_no_attack"] - variables["p1_no_attack"]
            

            # Count total moves with base_power > 0 for both
            p1_total_moves = len([m for m in variables["p1_base_powers"] if m > 0])
            p2_total_moves = len([m for m in variables["p2_base_powers"] if m > 0])
            features['p1_moves_advantage'] = p1_total_moves - p2_total_moves

            p1_high_damage_moves = sum(1 for d in variables["p1_damages"] if d > 0.3)  # >30% HP
            p2_high_damage_moves = sum(1 for d in variables["p2_damages"] if d > 0.3)  # >30% HP
            features['p1_high_damage_moves'] = p1_high_damage_moves
            features['p2_high_damage_moves'] = p2_high_damage_moves

            # Total number of moves by p1
            p1_moves_used = sum(1 for turn in timeline if turn.get('p1_move_details') is not None)
            # Total number of moves by p2
            p2_moves_used = sum(1 for turn in timeline if turn.get('p2_move_details') is not None)         
            features['moves_adv'] = p1_moves_used - p2_moves_used

            # Calculate the ratio of special and physical moves among all moves used
            p2_ratio = (variables["p2_special_moves"] + variables["p2_physical_moves"]) / p2_moves_used if p2_moves_used > 0 else 0
            p1_ratio = (variables["p1_special_moves"] + variables["p1_physical_moves"]) / p1_moves_used if p1_moves_used > 0 else 0
            # Difference between p1 and p2 ratios
            features['offensive_advantage'] = p1_ratio - p2_ratio

            features['p1_team_rm'] = team_size - ko_p1

            no_status_p1 = sum(
                1 for turn in timeline
                if turn.get('p1_pokemon_state', {}).get('status', 'nostatus') == 'nostatus'
            )
            no_status_p2 = sum(
                1 for turn in timeline
                if turn.get('p2_pokemon_state', {}).get('status', 'nostatus') == 'nostatus'
            )
            features["no_status_adv"] = no_status_p1 - no_status_p2

            status_p1 = sum(
                1 for turn in timeline
                if turn.get('p1_pokemon_state', {}).get('status', 'nostatus') != 'nostatus'
            )
            status_p2 = sum(
                1 for turn in timeline
                if turn.get('p2_pokemon_state', {}).get('status', 'nostatus') != 'nostatus'
            )

            features["status_adv"] = status_p1 - status_p2

            p1_total_boosts = [sum([variables["p1_atk_boosts"][i], variables["p1_def_boosts"][i], variables["p1_spa_boosts"][i], 
                    variables["p1_spd_boosts"][i], variables["p1_spe_boosts"][i]]) 
                for i in range(len(variables["p1_atk_boosts"]))]
            p1_std_total_boost = np.std(p1_total_boosts) if p1_total_boosts else 0

            p2_total_boosts = [sum([variables["p2_atk_boosts"][i], variables["p2_def_boosts"][i], variables["p2_spa_boosts"][i], 
                    variables["p2_spd_boosts"][i], variables["p2_spe_boosts"][i]]) 
                for i in range(len(variables["p2_atk_boosts"]))]
            p2_std_total_boost = np.std(p2_total_boosts) if p2_total_boosts else 0

            features["std_total_boost_adv"] = p1_std_total_boost - p2_std_total_boost

            # Difference between number of super effective moves of p2
            features['p2_super_effective'] =  variables["p2_super_effective"]
            features['p2_no_effective'] = variables["p2_no_effective"]
            
            # p1_final_hp: P1's percentage HP in the last turn
            final_turn = timeline[-1]
            p1_final_hp = final_turn.get('p1_pokemon_state', {}).get('hp_pct', 0)
            p2_final_hp = final_turn.get('p2_pokemon_state', {}).get('hp_pct', 0)
            features['final_hp_adv'] = p1_final_hp - p2_final_hp

            p1_avg_hp = np.std(variables["p1_list_hp"]) if variables["p1_list_hp"] else 0
            p2_avg_hp = np.std(variables["p2_list_hp"]) if variables["p2_list_hp"] else 0
            features['p1_std_hp_advantage'] = p1_avg_hp - p2_avg_hp
        
            # Battle ID and target variable
            features['battle_id'] = battle.get('battle_id')
            if 'player_won' in battle:
                features['player_won'] = int(battle['player_won'])
            
        feature_list.append(features)
    return pd.DataFrame(feature_list).fillna(0)