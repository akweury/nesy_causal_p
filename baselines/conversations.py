# Created by MacBook Pro at 02.09.25

def gpt_conversation(train_positive, train_negative, principle):
    conversation = [
        {
            "role": "user",
            "content": [
                # Provide example images
                {"type": "input_text", "text": "Here are three Positive examples:"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{train_positive[0]}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{train_positive[1]}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{train_positive[2]}"},

                {"type": "input_text", "text": "Here are three Negative examples:"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{train_negative[0]}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{train_negative[1]}"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{train_negative[2]}"},

                # Task instruction
                {"type": "input_text", "text": (
                    f"You are an AI reasoning about visual patterns using Gestalt principles.\n\n"
                    f"Principle under consideration: {principle}.\n\n"
                    "Based on the Positive and Negative examples, infer the logic rules "
                    "that distinguishes them.\n\n"
                    "Output ONLY the rules."
                )},
            ],
        }
    ]

    return conversation


def gpt_eval_conversation(image, logic_rules):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image}"},
                {"type": "input_text",
                 "text": f"Using the following reasoning rules: {logic_rules}. "
                         f"Classify this image as Positive or Negative."
                         f"Only answer with positive or negative."},
            ]
        }
    ]
    return conversation


def get_internVL_question(principle):
    background_knowledge = (
        "You are given images containing multiple objects and groups. "
        "Each object and group has attributes: shape, color, size, position, and group membership. "
        "Logical patterns in the image may involve single relations (e.g., all objects have the same color) "
        "or combinations of multiple relations (e.g., objects with the same shape are grouped together and mirrored along the x-axis). "
        "You can reason about: "
        "Individual attributes: shape, color, size, position; "
        "Group properties: number of members, grouping principle; "
        "Relations: same/different shape, color, size; mirrored positions; unique/diverse attributes within groups. "
        "Analyze the image by identifying both simple and complex combinations of these relations."
    )
    question = (
        f"{background_knowledge}\n\n"
        f"You are an AI reasoning about visual patterns using Gestalt principles.\n"
        f"Principle under consideration: {principle}.\n\n"
        f"We have a set of images labeled Positive and a set labeled Negative.\n"
        f"You will see each image one by one.\n"
        f"Observe each image, note any pattern features, and keep track of insights.\n"
        f"After seeing all images, we will derive the logic that differentiates Positive from Negative. "
        f"Please only answer with the logic/rule that distinguishes them.\n"
        f"Positive Images: Image 1: <image>, Image 2: <image>, Image 3: <image>. "
        f"Negative Images: Image 1: <image>, Image 2: <image>, Image 3: <image>."
    )
    return question

def llava_conversation(train_positive, train_negative, principle):
    background_knowledge = (
        "You are given images containing multiple objects and groups. "
        "Each object and group has attributes: shape, color, size, position, and group membership. "
        "Logical patterns in the image may involve single relations (e.g., all objects have the same color) "
        "or combinations of multiple relations (e.g., objects with the same shape are grouped together and mirrored along the x-axis). "
        "You can reason about: "
        "Individual attributes: shape, color, size, position; "
        "Group properties: number of members, grouping principle; "
        "Relations: same/different shape, color, size; mirrored positions; unique/diverse attributes within groups. "
        "Analyze the image by identifying both simple and complex combinations of these relations."
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": background_knowledge},
                {"type": "image", "image": train_positive[0]},
                {"type": "text", "text": f"You are an AI reasoning about visual patterns based on Gestalt principles.\n"
                                         f"Principle: {principle}\n\n"
                                         f"We have a set of images labeled Positive and a set labeled Negative.\n"
                                         f"You will see each image one by one.\n"
                                         f"Describe each image, note any pattern features, and keep track of insights.\n"
                                         f"After seeing all images, we will derive the logic that differentiates Positive from Negative. "
                                         f"The first positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[1]},
                {"type": "text", "text": f"The second positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_positive[2]},
                {"type": "text", "text": f"The third positive image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[0]},
                {"type": "text", "text": f"The first negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[1]},
                {"type": "text", "text": f"The second negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": train_negative[2]},
                {"type": "text", "text": f"The third negative image."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Now we have seen all the Positive and Negative examples. "
                                         "Please state the logic/rule that distinguishes them. "
                                         "Focus on the Gestalt principle of "
                                         f"{principle}."},
            ],
        },

    ]
    return conversation


def internVL_eval_question(logic_rules):
    return f"Using the following reasoning rules: {logic_rules}. Classify this image as Positive or Negative. Only answer with positive or negative. <image>\n"


def internVL_eval_conversation(image, logic_rules):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": f"Using the following reasoning rules: {logic_rules}. "
                         f"Classify this image as Positive or Negative."
                         f"Only answer with positive or negative."},
            ]
        }
    ]
    return conversation


def llava_eval_conversation(image, logic_rules):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": f"Using the following reasoning rules: {logic_rules}. "
                         f"Classify this image as Positive or Negative."
                         f"Only answer with positive or negative."},
            ]
        }
    ]
    return conversation



