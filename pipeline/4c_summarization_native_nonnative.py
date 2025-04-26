import csv
import os
import pandas as pd
from helpers import get_gpt_response
import re

# ——— CONFIG ———
input_file  = 'processed_output/output_translated_cleaned.csv'
output_file = 'processed_output/output_summarized_by_type.csv'
# —————————

df = pd.read_csv(input_file)
df.dropna(subset=['translation'], inplace=True)

# group all rows by image
grouped = df.groupby('image_link')

# keep track of which (image, type) we've already done
if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
    done = pd.read_csv(output_file)
    completed = set(zip(done['image_link'], done['annotation_type']))
else:
    completed = set()

with open(output_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # header: image_link, annotation_type (native/nonnative), ids, transcriptions, distinct, explanations, summary
    if os.stat(output_file).st_size == 0:
        writer.writerow([
            'image_link',
            'annotation_type',
            'ids',
            'languages',
            'transcriptions_translated',
            'culturally_distinct',
            'culturally_distinct_explanations',
            'summary'
        ])

    for image_link, group in grouped:
        pattern = re.compile(r'.*/([^/]+)_images/')  # match last instance of _images
        m = pattern.search(image_link)
        source_lang = m.group(1) if m else ''
        # split into native vs nonnative
        native     = group[group['language'] == source_lang.capitalize()]
        nonnative  = group[group['language'] != source_lang.capitalize()]

        for annotation_type, subgrp in (('native', native), ('nonnative', nonnative)):
            if subgrp.empty:
                continue

            key = (image_link, annotation_type)
            if key in completed:
                print(f"Skipped {image_link} ({annotation_type})")
                continue

            ids           = subgrp['id'].tolist()
            languages     = subgrp['language'].tolist()
            transcriptions = subgrp['translation'].tolist()
            distinct      = subgrp['culturally_distinct'].tolist()
            explanations  = subgrp['cultural_distinction_explanation'].tolist()

            if len(transcriptions) > 1:
                messages = [
                    {
                        'role': 'system',
                        'content': "Enhance the quality of the raw captions into a high-quality transcript. Provide clear, descriptive, and professional outputs. Please make sure to include every single detail from every transcription; do not be too concise. Avoid rewording things if it makes the output more vague. If there are any culturally distinct elements, please explain them in detail. Also, it's okay if the output is longer than the input. It is important that you do not include any other text or formatting besides the summary itself. For example, do not include 'Summary:', 'The transcription describes', 'The first transcription mentions' or any other meta text; it is critical that your only output is the caption with all of the details from the original caption(s)."
                    },
                    # {
                    #     'role': 'user',
                    #     'content': """Transcriptions: ["This is a picture of a long black bird with a lot of iridescent accents. It looks like an iridescent blue accenting on the wing and also maybe an iridescent purple there and then going up the back of its head to the calf and then on its chest and maybe even underneath. I'm pretty sure I've seen a bird like this in St. Lucia. It has piercing yellow eyes, dark long sharp colored beak. It's pretty dark legs and it seems to be perched on what is like a cement wall. It looks like some sort of man-made wall that has some red staining on it like residue of something. In the background you could see blurry but there appear to be some sort of tropical plants. They all have like broad palm-like leaves and maybe there's a tall tree structure in the center behind the bird but the bird is looking off to the right. It has a very long tail and very sharp looking features, sharp wings, sharp tail and a sharp beak almost like it's a fighter plane, this kind of bird.", "We've heard resting on a cement center divider. It's technically not, well, from where we're looking, it might not even be in our road. It might just be that kind of cement piece that's just somewhere else, like yet to be used. The bird itself is blue, kind of long. Long tail feather, long beak, yellow eye ring, also some strange red paste-like material under the bird. I'm not sure if it's blood or maybe there's some berries up above on the tree. The bird's just kind of hanging out.", "This is a photograph of a black bird. This bird is very thin and long and he is perched on, it looks like a cement barrier or a cement wall, so he's perched on top of that and he is looking to the top right corner of the picture. In the background of the picture, it's very blurry, there's some greens and whites, it's very blurred out, it looks like it could be greenery in the back. On the wall that this bird is sitting on it's mostly white with some small specks of red or pink and white and light blues in this wall that he's standing on. The bird has long skinny legs and short talons and it's got a very long tail that's pointed down to the left of the picture. All of his feathers are black, the breast and the wings and the tail and the head and everything is black with a little bit of shine in the wings to give it kind of an iridescent green color on the wing. It's got a yellow eye and it's got kind of a medium to long length beak that's thin and that's pointed up to the top and right hand side of the picture.", "This is a photograph of a black bird sitting on a concrete divider. There's green in the background, which is leaves and grass. The bird looks like a crow and has a yellow eye with a black body. It also has a long tail. The concrete divider looks like it's part of a building or maybe a parking lot, and the bird's talons are wrapped around the side of it and it's perched up. The bird also has quite a long beak, and the photograph is outside with the colors white, green, black, and it looks like the wings are a bit iridescent of the bird. And the bird is facing to the left.", "This is a picture of a bluebird with a short gray beak. It almost looks like a thin raven. It's got a blue to black coloring, some light blue on the feathers. The edges of the feathers are black. The tail is of the darker color. It has two claws. The claws are gripping a white concrete structure, almost like a patio to a building, and it is just perched on this white structure. This wall, top of a wall perhaps, looking out in the background, everything is blurred because it's a close-up of only the bird. So the background has blurred greenery, some white, some brown, but mostly green from leaves or trees in the background. And it has a yellow eye, it has a gray beak, and it appears to be somewhat of a very thin-looking raven.", "This is a photograph of a blackbird. The blackbird has a dark blue color in its feathers and a long tail behind it. The bird is looking up towards the upper right-hand corner of the frame of the photo. The bird is sitting on a white wall that has red splatters on it. It's a sunny day in the photo. There's greenery visible but blurred in the background and the trunk of a white tree visible. The bird's eye is yellow and its feathers are shiny."]"""
                    # },
                    # {
                    #     'role': 'assistant',
                    #     'content': """This photograph depicts a striking black bird, possibly a grackle or similar species, perched on a white cement wall with red stains. The bird's sleek, elongated body is adorned with iridescent feathers that shimmer with shades of blue, purple, and green, most prominently on its wings and back. Its piercing yellow eye and long, sharp beak lend it a fierce, almost aerodynamic appearance. The bird's dark, slender legs and short talons grip the edge of the cement structure, which resembles a divider or barrier, possibly in an outdoor setting like a park or building patio. The background is a blur of green and white hues, hinting at lush tropical plants and tall trees, setting a serene, natural scene. The bird is poised, looking towards the upper right-hand corner of the frame, with its long tail feathers trailing elegantly to the left."""
                    # },
                    # {
                    #     'role': 'user',
                    #     'content': """Transcriptions: ["This is an image of a logo. The background of the logo is just black, a black panel. And then the logo is - it almost looks like the PlayStation logo with the P and the S, the P standing up and the S on the ground, and then the registered trademark to the right. And that's in white in the dead center. And below it, it reads in all caps, VIDEO GAME COVER ART, and kind of just again an all-cap basic font. And that is the totality of this image. Again, this white logo component, the actual logo and the text, and then all around that you just have a black background. So maybe this is some sort of file that is given to a designer for some sort of project that requires the logo.", "This is an image of a logo. It's on a black background. This looks like the PlayStation logo. It's a P standing up and an S laying down. It's a white logo. Under that it says in all white capital letters, Video Game Cover Art, and it is a looks like it's written in bold, like a bold thick font, like Helvetica font. The background is a large black rectangle and it's just all black and it's just this PlayStation logo and these white words across the middle.", "In this picture is a basic black background, and you see little streaks of gray interspersed in there somewhat, just it's not completely solid black, but in the background you see a little square box that's a little bit lighter color of black, and then you see, it looks like a letter P that is upright, and then a letter S that is down, like flat on the ground, but it's in white text, and a little R symbol, it might be for PlayStation PS, and then down below that, that was centered in the center of the picture, down below that it says Video Game, and the next line with the all capital letters, and then underneath that it says Cover Art, in all capital letters again, all those letters are centered, and the PS is angled a little bit, like as if it's rotated, the P is rotated as if it's standing on end, it's rotated clockwise about 45 degrees."]"""
                    # },
                    # {
                    #     'role': 'assistant',
                    #     'content': """The image features a logo on a predominantly black background. The background isn't a solid black but has subtle streaks of gray interspersed throughout. Centered in this black expanse is a white logo that bears a striking resemblance to the PlayStation logo, with a stylized 'P' standing upright and an 'S' laying flat beneath it. This 'PS' logo is slightly angled, as if the 'P' is rotated clockwise by around 45 degrees. To the right of this logo is a registered trademark symbol '®'. Beneath the logo, the phrase "VIDEO GAME COVER ART" is displayed in all capital letters, written in a bold, straightforward font, possibly Helvetica. All the text elements are centered within the image, suggesting that this might be a preliminary design file intended for a designer working on a project related to video game cover art."""
                    # },
                    {
                        'role': 'user',
                        'content': f"Transcriptions: {transcriptions}\n{f"Extra explanation: {explanations}" if explanations else ''}"
                    }
                ]
            else:
                messages = [
                    {
                        'role': 'system',
                        'content': "Enhance the quality of the raw caption into a high-quality transcript. Provide clear, descriptive, and professional outputs. Please make sure to include every single detail from the transcription; do not be too concise. Avoid rewording things if it makes the output more vague. If there are any culturally distinct elements, please explain them in detail. Also, it's okay if the output is longer than the input. It is important that you do not include any other text or formatting besides the summary itself. For example, do not include 'Summary:', 'The transcription describes', 'The first transcription mentions' or any other meta text; it is critical that your only output is the caption with all of the details from the original caption."
                    },
                    # {
                    #     'role': 'user',
                    #     'content': """Transcription: \"This is a large brown suckerfish that is inside of a home fish tank. So, this is a type of catfish, but what it does, it goes along in any surface that it can get its mouth around. It kind of just sucks onto it with its mouth, which is a big sucker-type mouth, and it eats all of the algae or whatever other type of debris that has nutrition in it. So, they're good for fish tanks because they keep the water looking very clean-looking. They will attach to the sides of the fish tank, the glass, they will attach themselves to that and go up and down the sides of fish tanks, cleaning the glass, removing all the impurities, and that's how they get their nutrients to survive. But right now, it looks like it's attached to a piece of driftwood that is inside of this aquarium, and then all around it, there are fake plants planted in the bottom of this aquarium, and it looks like the bottom of the aquarium is some black rocks. This is a black suckerfish, which is a type of catfish often seen in people's home aquariums because they are very good for keeping the home aquariums clean, the water clean inside of them \""""
                    # },
                    # {
                    #     'role': 'assistant',
                    #     'content': """A large brown suckerfish, commonly known as a plecostomus or pleco catfish, is prominently displayed inside a home fish tank. This species, recognized for its sucker-like mouth, clings to various surfaces within the tank, feeding on algae and other nutritional debris. This natural cleaning behavior helps maintain the tank's cleanliness, making it a popular choice among aquarium enthusiasts. In the image, the suckerfish is currently affixed to a piece of driftwood. Surrounding it are artificial plants anchored at the base of the aquarium, which is lined with black rocks. The plecostomus, often seen in household aquariums, contributes to the pristine condition of the tank by continuously scouring the glass and other surfaces, ensuring a clean and healthy environment."""
                    # },
                    # {
                    #     'role': 'user',
                    #     'content': """Transcription: "It's a conference room at Barrett Jackson, which is an automotive dealer, and this conference room is very unique because the conference table, the very centerpiece of the conference table is a Porsche 911. This is probably like a 70s-style Porsche 911, and then around the Porsche 911 has been installed a large clear glass tabletop, which makes up a conference room table. Around this conference room table are several office chairs. They look like they are made from racing seats that would have been taken out of a race car and then transformed into an office chair with roller wheels on them and armrests. And on the back wall, you can see the logo from Barrett Jackson. So this is obviously the Barrett Jackson museum or building where they do their car auctions at or maybe the corporate offices, but it's from Barrett Jackson. This is one of their conference rooms that has been built around a Porsche 911, a vintage Porsche 911, which is black. The decorations in the room are red, white, and black. Red lighting, red accents, white panels on the walls. The Porsche 911 is black itself, and it has red neon lights lighting up the underneath of it.\""""
                    # },
                    # {
                    #     'role': 'assistant',
                    #     'content': """In the Barrett-Jackson corporate office, an extraordinary conference room captures the essence of automotive passion and innovation. Dominating the room is a unique conference table, ingeniously incorporated around a vintage black 1970s Porsche 911. The classic sports car, sitting as the centerpiece, is encased in a large clear glass tabletop, transforming it into a functional yet remarkable discussion surface. Surrounding this one-of-a-kind table are several office chairs fashioned from racing seats, complete with roller wheels and armrests, epitomizing the fusion of comfort and high-octane style. The room's design boasts a color scheme of red, white, and black, with striking red lighting and accents complementing the white wall panels. The back wall prominently features the Barrett-Jackson logo, reiterating the prestigious automotive dealer's identity. Underneath the car, red neon lights cast a captivating glow, accentuating the vintage Porsche's sleek lines and reinforcing the room's thematic cohesion. This conference room is a testament to Barrett-Jackson's dedication to automotive excellence and distinctive style."""
                    # },
                    {
                        'role': 'user',
                        'content': f"Transcriptions: {transcriptions[0]}\n{f"Extra explanation: {explanations[0]}" if explanations else ''}"
                    }
                ]

            try:
                summary = get_gpt_response(messages)
                # flatten any newlines
                summary = summary.replace("\n", " ").replace("\r", " ")
                writer.writerow([
                    image_link,
                    annotation_type,
                    ids,
                    languages,
                    transcriptions,
                    distinct,
                    explanations,
                    summary
                ])
                f.flush()
                print(f"➔ Summarized {image_link} ({annotation_type})")
            except Exception as e:
                print(f"Error on {image_link} ({annotation_type}): {e}")
