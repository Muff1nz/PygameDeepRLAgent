from cmd import Cmd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

class EMinspector(Cmd):
    version = "0.9"
    memoryPath = 'C:/deepRLAgent/Memory/'
    agentName = "dqn_1"

    processedFrames = None
    experienceMemory = None
    replayMetaFile = None

    def do_loadMemory(self, line):
        processedFramesFile = self.memoryPath + self.agentName + "_" + \
                              self.version + "_processedFrames_" + line + ".npy"
        experienceMemoryFile = self.memoryPath + self.agentName + "_" + \
                               self.version + "_experienceMemory_" + line + ".npy"
        replayMetaFile = self.memoryPath + self.agentName + "_" + \
                         self.version + "_replayMeta_" + line

        self.processedFrames = np.load(processedFramesFile)
        self.experienceMemory = np.load(experienceMemoryFile)
        self.replayMetaFile = open(replayMetaFile, 'r')

        print("Current ei: {}EM full: {}\n".format(
            self.replayMetaFile.readline(), self.replayMetaFile.readline()
            )
        )

    def do_displayMemory(self, line):
        experience = self.experienceMemory[int(line)]
        img = self.processedFrames[int(line)]
        print("Action: {}, Reward: {}".format(experience[0], experience[1]))
        plt.imshow(img, cmap='gray')
        plt.show()

    def do_status(self, line):
        print("Path: {}".format(self.memoryPath))
        print("Version: {}".format(self.version))
        print("Agent name: {}".format(self.agentName))

    def do_setVersion(self, line):
        self.version = line

    def do_setMemoryPath(self, line):
        self.memoryPath = line

    def do_setAgentName(self, line):
        self.agentName = line


if __name__ == '__main__':
    EMinspector().cmdloop()