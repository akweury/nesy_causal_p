# Created by X at 12.03.25
import os
import beta_config

def save_results(final_tokens):
    overall_save_path = os.path.join(beta_config.OUTPUT_DIR, "final_tokens.pkl")


    print(f"Final tokens saved to {overall_save_path}")
