# Enhanced Embeddings in Zero-Shot Learning for Environmental Audio

## Partition Strategies

Two partition strategies are used for [ESC-50](https://github.com/karolpiczak/ESC-50), from [Zero-Shot Audio Classification via Semantic Embeddings](https://arxiv.org/abs/2011.12133).

**Category partition**

|                Fold                | Classes                                                                                                                                                |
| :--------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
|           Animal Sounds            | dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow                                                                                           |
| Natural Soundscapes & Water Sounds | rain, sea waves, crackling fire, crickets, chirping birds, water drops, wind, pouring water, toilet flush, thunderstorm                                |
|     Human (Non-Speech) Sounds      | crying baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing teeth, snoring, drinking sipping                                   |
|      Interior/Domestic Sounds      | door wood knock, mouse click, keyboard typing, door wood creaks, can opening, washing machine, vacuum cleaner, clock alarm, clock tick, glass breaking |
|       Exterior/Urban Sounds        | helicopter, chainsaw, siren, car horn, engine, train, church bells, airplane, fireworks, hand saw                                                      |

**Random Partition**

| Fold | Classes                                                                                                                   |
| :--: | :------------------------------------------------------------------------------------------------------------------------ |
|  1   | brushing teeth, church bells, clock tick, cow, drinking sipping, fireworks, helicopter, mouse click, pig, washing machine |
|  2   | clapping, crickets, glass breaking, hand saw, keyboard typing, laughing, siren, sneezing, thunderstorm, vacuum cleaner    |
|  3   | breathing, chainsaw, chirping birds, coughing, door wood creaks, door wood knock, frog, pouring water, rain, train        |
|  4   | airplane, can opening, crying baby, engine, footsteps, hen, insects, rooster, snoring, toilet flush                       |
|  5   | car horn, cat, clock alarm, crackling fire, crow, dog, sea waves, sheep, water drops, wind                                |

## Word Embeddings

Often, the input to the word embedding model is the class label. This work considers synonyms, semantic broadening and onomatopoeia. These are the words for each class.

| Class label        | Added words                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| dog                | dog, canine, bark, woof, yap, call, animal, puppy                                               |
| rooster            | rooster, cockerel, call, animal                                                                 |
| pig                | pig, hog, sow, swine, squeal, oink, grunt, call, animal                                         |
| cow                | cow, moo, call, bull, oxen, animal                                                              |
| frog               | frog, toad, croak, call, animal                                                                 |
| cat                | cat, meow, mew, purr, hiss, chirp, kitten, feline, call, animal                                 |
| hen                | hen, cluck, chicken, animal, call                                                               |
| insects, flying    | insects, flying, buzz, hum, bug                                                                 |
| sheep              | sheep, bleat, animal, call, lamb                                                                |
| crow               | crow, squawk, screech, caw, bird, call, cry, animal                                             |
| rain               | rain, drizzle, wet, sprinkle, shower, water, nature                                             |
| sea, waves         | sea, waves, water, swell, tide, ocean, surf, nature                                             |
| crackling, fire    | crackling, fire, hissing, sizzling, flame, bonfire, campfire, nature                            |
| crickets           | crickets, insects, insect, bug, cicada, call                                                    |
| chirping, birds    | chirping, birds, animal, call, song, tweet, chirp, twitter, trill, warble, chatter, cheep       |
| water, drops       | water, drops, splash, droplet, drip                                                             |
| wind               | wind, nature, gust, gale, blow, breeze, howl                                                    |
| pouring, water     | pouring, water, slosh, gargle, splash, splosh                                                   |
| toilet, flush      | toilet, flush, water, flow, wash                                                                |
| thunderstorm       | thunderstorm, thunder, storm, nature, lightning                                                 |
| crying, baby       | crying, baby, cry, human, whine, infant, child, wail, bawl, sob, scream, call                   |
| sneezing           | sneezing, sneeze                                                                                |
| clapping           | clapping, clap, applause, applaud, praise                                                       |
| breathing          | breathing, breath, breathe, gasp, exhale                                                        |
| coughing           | coughing, cough, hack                                                                           |
| footsteps          | footsteps, walking, walk, pace, step, gait, march                                               |
| laughing           | laughing, cackle, laugh, chuckle, giggle, funny                                                 |
| brushing, teeth    | brushing, teeth, scrape, rub, brush                                                             |
| snoring            | snoring, snore, sleep, snore, snort, wheeze, breath                                             |
| drinking, sipping  | drinking, sipping, gulp, gargle, drink, sip, breath                                             |
| door, knock        | door, wood, knock, tap, bang, thump                                                             |
| mouse, click       | mouse, click, computer, tap                                                                     |
| keyboard, typing   | keyboard, typing, tap, mechanical, computer                                                     |
| door, wood, creaks | door, wood, creaks, squeak, creak, screech, scrape                                              |
| can, opening       | can, opening, hiss, fizz, air                                                                   |
| washing, machine   | washing, machine, electrical, hum, thump, noise, loud                                           |
| vacuum, cleaner    | vacuum, cleaner, electrical, noise, loud                                                        |
| clock, alarm       | clock, alarm, signal, buzzer, alert, ring, beep                                                 |
| clock, tick        | clock, tick, tock, click, clack, beat, tap, ticking                                             |
| glass, breaking    | glass, breaking, crunch, crack, smash, clink, break, noise                                      |
| helicoper          | helicoper, chopping, engine, blades, whirring, swish, chopper, electrical, noise, vehicle, loud |
| chainsaw           | chainsaw, saw, electrical, noise, tool, loud                                                    |
| siren              | siren, alarm, alert, bell, horn, noise, loud                                                    |
| car, horn          | car, horn, vehicle, noise, blast, loud, honk                                                    |
| engine             | engine, rumble, vehicle, chug, revving, car, drive                                              |
| train              | train, clack, horn, clatter, vehicle, squeal, rattle                                            |
| church, bells      | church, bells, tintinnabulation, ring, chime, bell                                              |
| airplane           | airplane, plane, motor, engine, hum, loud, noise                                                |
| fireworks          | fireworks, burst, bang, firecracker                                                             |
| hand, saw          | hand, saw, squeak, sawing, cut, hack, tool                                                      |
