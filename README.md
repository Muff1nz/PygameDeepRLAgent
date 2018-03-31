I've been updating this readme as i experiment and make changes to the code, which could include changes to the actual neural network. This means that trying to reproduce these results now by using the parameters i used might not give the same results, because the network is likely different now from when i ran the training session that produced the result given in the readme. Looking at the version control history of this readme, and checking out the commit that the result was added in should work.

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
![a3cshootinggroundslr](https://user-images.githubusercontent.com/29259118/30173297-77c11ccc-93f7-11e7-87f6-fc83e60e3070.PNG)
![a3cshootinggroundsscore](https://user-images.githubusercontent.com/29259118/29883477-beb05af0-8db0-11e7-8080-269f952695ac.PNG)

The above images shows the score and learning rate per episode of 8 A3C worker agents during their almost 18 hour training session in the ShootingGrounds level of A3CBootcamp. The agents control a blue square with the ability to shoot, and it has to shoot the read squares. Shooting a red square rewards the agent with 1 point. The agent needs to shoot as many red squares as possible within the time limit to get the most points.

Youtube video of agent progress in ShootingGrounds:
https://www.youtube.com/watch?v=fEKITU7cjNg&feature=youtu.be

# Causality tracking

Causality tracking is a system in this project that tries to solve the credit assignment problem. Causality tracking assigns rewards to the (action, state) tuple that caused the reward. In practice this means that the game keeps track of at which time step all bullets are fired, and when a bullet hits something, the reward is credited to the (action, state) tuple from which the bullet was fired instead of the most current (action, state) tuple.

## Test
This causality tracking test was done in the ShootingGrounds game.

![shootinggroundscttestlr](https://user-images.githubusercontent.com/29259118/30173301-7b984622-93f7-11e7-9612-3c1df33d79de.PNG)

The above image shows the learning rate for both test.
Both test were run for 10K episodes with 16 worker agents, all hyper parameters were the same.

### Causality tracking disabled
![shootinggroundsctfalsescore](https://user-images.githubusercontent.com/29259118/30173306-7d098b9c-93f7-11e7-9e1f-7ea30df185eb.PNG)

### Causality tracking enabled
![shootinggroundscttruescore](https://user-images.githubusercontent.com/29259118/30173308-7e37c1f0-93f7-11e7-8729-2a6b686ca35b.PNG)

## Result
With causality tracking disabled, the agent performance peaked at 20 points, with causality tracking enabled performance peaked at 25 points.

This means that for this experiment causality tracking improved perfomance by 25%



