def run(settings, gameDataQueues, playerActionQueues): # run this as the target of a process
    import asyncio

    if (len(gameDataQueues) != len(playerActionQueues)):
        raise RuntimeError("GameRunner inputs gameDataQueues and playerActionQueues are not the same length!")

    game = settings.games[settings.game][0]
    games = []
    for gameData, playerAction in zip(gameDataQueues, playerActionQueues):
        games.append(game(settings, gameData, playerAction))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(*(game.run() for game in games)))
    # the below line wont really ever be reached, this process will just be terminated by the trainer when its done
    loop.close()


