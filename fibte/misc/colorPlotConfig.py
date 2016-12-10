import random
import os
import pickle
import subprocess

from fibte import COLORS_CONFIG

tableau10 = [(23, 190, 207), # Blau clar
             #(31, 119, 180), # Blau fosc
             #(44, 160, 44),  # Verd
             (127, 127, 127),# Gris
             (140, 86, 75),  # Marro
             (148, 103, 189),# Lila
             (188, 189, 34), # Caqui
             #(214, 32, 40),  # Vermell
             #(227, 119, 194),# Rosa
             (255, 127, 14), # Taronja
             ]

class ColorPlotConfig(object):
    """Stores colors of things that are plotted by name
    """
    def __init__(self):
        self.allColors = self._getAllColors()
        self.loadColors()
        # Bad fix here to manually set the colors...
        self.colors['ECMP'] = (44/255., 160/255., 44/255.)  # Verd
        self.colors['Ideal'] = (214/255., 32/255., 40/255.) # Vermell
        self.colors['EDS-Best'] = (31/255., 119/255., 180/255.) # Blau fosc
        self.colors['MDS'] = (227/255., 119/255., 194/255.)# Rosa

    def _getAllColors(self):
        colorset = set()
        for i in range(len(tableau10)):
            r, g, b = tableau10[i]
            colorset.add((r / 255., g / 255., b / 255.))
        return colorset

    def loadColors(self):
        """Loads the stored hash seeds from the file"""
        self.availableColors = set()
        if os.path.isfile(COLORS_CONFIG):
            with open(COLORS_CONFIG,"r") as f:
                self.colors = pickle.load(f)
        else:
            self.colors = {}

        # Update available colors
        picked_colors = set(self.colors.values())
        self.availableColors = self.allColors.difference(picked_colors)

    def saveColors(self):
        """Stores the hash seeds in the file"""
        with open(COLORS_CONFIG+"_tmp", 'w') as f:
            pickle.dump(self.colors, f)
        subprocess.Popen(["mv", "{0}_tmp".format(COLORS_CONFIG), COLORS_CONFIG])

    def getColor(self, name):
        """Gets the current seed of the node defined by name"""
        color = self.colors.get(name, None)
        if color:
            return color
        else:
            color = self.getAvailableColor()
            self.setColor(name, color)
            self.saveColors()
            return color

    def setColor(self, name, color):
        """Checks whether the host is in the seeds database. If it isn't, we add it.
        It does not rewrite the seed, but updates the database.
        """
        self.colors[name] = color
        self.saveColors()

    def getAvailableColor(self):
        """"""
        if self.availableColors:
            return random.choice(list(self.availableColors))
        else:
            raise ValueError("No more available colors!")