# PygameDeepRLAgent
This project is about training deepRL agents at varius tasks made with pygame. Currently using an A3C agent.

# Results
## FeedingGrounds
![alt text](https://user-images.githubusercontent.com/29259118/29706944-cc96acae-8983-11e7-9b85-ffa41f7a8fae.PNG)

The above image shows score per episode for 8 workers during their 1 day and 20 hour training session in the A3Cbootcamp game level FeedingGrounds, a game where the agent has to "eat food" by moving to the green squares, the agent controls a blue square inside a square environment.

The agent was trained using a i7 6700k and a GTX 1080 ti

![a3c_0 97_plus_18900](https://user-images.githubusercontent.com/29259118/29707286-0f5ffc24-8985-11e7-8b04-76d363726d85.gif)

The above gif shows a sequence of the game, the way the agent sees it.

## ShootingGrounds
![a3cshootinggroundsscore](https://user-images.githubusercontent.com/29259118/29883477-beb05af0-8db0-11e7-8080-269f952695ac.PNG)

The above image shows the score per episode of 8 A3C worker agents during their almost 18 hour training session in the ShootingGrounds level of A3CBootcamp. The agents control a blue square with the ability to shoot, and it has to shoot the read squares. Shooting a red square rewards the agent with 1 point. The agent needs to shoot as many red squares as possible within the time limit to get the most points.

Youtube video of agent progress in ShootingGrounds:
https://www.youtube.com/watch?v=fEKITU7cjNg&feature=youtu.be
