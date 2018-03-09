# Notes on project meeting 1 with Prof Drori:

## Some general aspects:

- The main aspect of the project: cooperative game, not only adversarial.
- Multiple agents towards a common goal.
- Don't know the MDP, the envt...

- Do our agents decide **on their own** or is there a **centralized policy** ? What will be centralized, and what will be decentralized ?

## About COMA:

- Each agent learns its policy (with methods similar to what was seen in class), yet we need a *central critic* to judge each agent's action.
- The presentation on COMA is available online (YouTube ?).
- Drori thinks that COMA is good and advises us to check if the code is available online.

## BiCNET:

- It's good too, check it out. 
- If BiCNET has online code and not COMA, might be better to start working with this one.

## MLSH:

- Multiple policies for each unit
- Therefore each policy can switch policy
- Might be a good approach because it's similar to what humans do when they play.
- In particular, can be useful for our harder minigame where we have different opponents with different skills.

## TODO:

- Checkout BiCNET and MLSH.
- In "Our Approach" section: Define more precisely (mathematically) our setting: the multi-agent setting, the assumptions.
- Explain how BiCNET/COMA fall into these settings
- Explain that we will try to add MLSH on top of COMA and BiCNET, which hasn't been done before by other research groups.

## Resources:

- TODO: find BiCNET and MLSH resources
- njustesen.com
- Find the video on COMA