import pandas as pd

def generate_lists():
    """
    Generates two lists: one with very short recipes and another with very short job descriptions.
    
    Returns:
        tuple: A tuple containing two lists, one for recipes and one for job descriptions.
    """
    recipes = [
    "Tomato Soup: Blend canned tomatoes, broth, season, and simmer.",
    "Grilled Cheese: Butter bread, add cheese, grill until golden.",
    "Pasta Salad: Cook pasta, mix with veggies and dressing.",
    "Stir Fry: Sauté veggies and protein, add sauce, serve over rice.",
    "Omelette: Whisk eggs, pour into pan, fill, and fold.",
    "Smoothie: Blend fruit, yogurt, and ice until smooth.",
    "Pancakes: Mix batter, pour on griddle, flip when bubbly.",
    "Garlic Bread: Spread garlic butter on bread, bake until crisp.",
    "Avocado Toast: Mash avocado, spread on toast, add toppings.",
    "Cereal: Pour cereal in bowl, add milk.",
    "BLT Sandwich: Layer bacon, lettuce, tomato, and mayo on bread.",
    "Mug Cake: Mix ingredients in mug, microwave for 90 seconds.",
    "Nachos: Layer chips, cheese, beans, bake, add toppings.",
    "Mac and Cheese: Cook pasta, stir in cheese sauce.",
    "Tuna Salad: Mix canned tuna with mayo and celery.",
    "French Toast: Dip bread in egg mixture, cook until golden.",
    "Baked Potato: Pierce potato, bake, and add toppings.",
    "Quesadilla: Fill tortilla with cheese, fold, and grill.",
    "Fruit Salad: Chop fruit, toss with juice, chill.",
    "Guacamole: Mash avocados, mix with onion, tomato, lime.",
    "Bruschetta: Top sliced bread with tomato mixture, broil.",
    "Pesto Pasta: Cook pasta, mix with pesto sauce.",
    "Hummus: Blend chickpeas, tahini, lemon, garlic, serve with veggies.",
    "Caprese Salad: Layer tomato, mozzarella, basil, drizzle balsamic.",
    "Salsa: Dice tomatoes, onions, peppers, mix with lime and cilantro.",
    "Greek Salad: Combine cucumber, tomato, onion, feta, olives.",
    "Chili: Cook meat, beans, tomatoes, spices, simmer.",
    "Egg Salad: Chop boiled eggs, mix with mayo, mustard, spices.",
    "Sliders: Form small patties, grill, serve on mini buns.",
    "Popcorn: Pop kernels, add butter and salt."
    ]

    job_descriptions = job_descriptions = [
    "Software Developer: Writes and tests computer code.",
    "Graphic Designer: Creates visual content using software.",
    "Teacher: Educates students and prepares lesson plans.",
    "Nurse: Provides patient care and administers medication.",
    "Accountant: Manages financial records and tax filings.",
    "Electrician: Installs and repairs electrical systems.",
    "Chef: Prepares meals and designs menus.",
    "Lawyer: Represents clients in legal matters.",
    "Salesperson: Sells products and negotiates contracts.",
    "Journalist: Reports news and conducts interviews.",
    "Architect: Designs buildings and oversees construction.",
    "Engineer: Solves problems using scientific principles.",
    "Dentist: Diagnoses and treats oral health issues.",
    "Pharmacist: Dispenses medication and advises on drug therapy.",
    "Carpenter: Constructs and repairs building frameworks.",
    "Plumber: Installs and maintains water systems.",
    "Mechanic: Diagnoses and repairs vehicles.",
    "Photographer: Takes and edits photographs.",
    "Veterinarian: Provides medical care to animals.",
    "Pilot: Operates aircraft and navigates flights.",
    "Cashier: Handles cash transactions and customer service.",
    "Librarian: Manages library resources and assists patrons.",
    "Security Guard: Protects property and maintains safety.",
    "Bartender: Mixes drinks and manages bar area.",
    "Fitness Trainer: Leads exercise sessions and advises clients.",
    "Translator: Converts written text between languages.",
    "Web Developer: Builds and maintains websites.",
    "Nutritionist: Advises on diet and food choices.",
    "Real Estate Agent: Helps clients buy, sell, and rent properties.",
    "Customer Service Rep: Assists customers and resolves issues."
    ]

    return recipes, job_descriptions

def create_dataframe():
    """
    Generated a list of recipes and job descriptions and creates and shuffles a Dataframe with them.

    Returns:
        DataFrame: A shuffled pandas DataFrame with text and corresponding labels.
    """
    recipes, job_descriptions = generate_lists()
    labels = [0] * len(recipes) + [1] * len(job_descriptions)
    texts = recipes + job_descriptions
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df.sample(frac=1).reset_index(drop=True)



