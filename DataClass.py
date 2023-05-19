import numpy as np
import math

class Data():
    #""
        #Implementation of the data storage.
    #"""

    def __init__(self, initial_data):
        self.POWER = 10 #[W]
        self.RXGAIN = 30 #[dB]
        self.RXLOSS = 1 #[dB]
        self.TXGAIN = 30 #[dB]
        self.TXLOSS = 3 #[dB]
        self.FREQUENCY = 4e12 #[Hz]
        self.BANDWIDTH = 1e6 #[Hz]
        self.SYMBOLRATE = 1e3 #[Hz]
        self.MODULATIONORDER = 4 #QPSK
        self.SENSITIVITY = -160 #[dBW]
        #self.DATATOSEND = 1e6 #[bits]
        #self.AVAILABLEENERGY = 5000 #[J]
        self.current_data = initial_data

    #def remove_data(self, data):
     #   """ Removes data from sorage """
      #  self.current_data = self.current_data - data

       # if self.current_data < 0:
        #    self.current_data = 0

    def reset_data_storage(self, initial_data): #Joule
        """ Resets current_data to intial value """
        self.current_data = initial_data
    
        
    
    #'''
    #    CALCULATION FUNCTIONS
    #'''
    
    def calculateFreeSpaceLoss(self, dist):
        fsllinear = (4 * np.pi * dist * self.FREQUENCY / 3e8)**2
        return 10*np.log10(fsllinear)
    
    def calculateNoise(self):
        return 10*np.log10(1.38e-23 * 290 * self.BANDWIDTH) #kTB assuming temperature of 290K (17C)
    
    def calculateSNR(self, dist):
        return (10*np.log10(self.POWER) + self.RXGAIN + self.TXGAIN - self.RXLOSS - self.TXLOSS - self.calculateFreeSpaceLoss(dist)) - self.calculateNoise()
    
    def calculateIdealDataRate(self, dist):
        # This is the Shannon-Hartley Theorem
        return self.BANDWIDTH * np.log2(1 + 10**(self.calculateSNR(dist)/10))
    
    def calculateBER(self, dist):
        bitrate = self.SYMBOLRATE * np.log2(self.MODULATIONORDER)
        efficiency = bitrate / self.BANDWIDTH
        ebn0 = 10**(self.calculateSNR(dist)/10) / efficiency
        return (1/np.log2(self.MODULATIONORDER)) * math.erfc(np.sqrt(2*ebn0))
    
    def calculateEffectiveDataRate(self, dist):
        receivedPower = (10*np.log10(self.POWER) + self.RXGAIN + self.TXGAIN - self.RXLOSS - self.TXLOSS - self.calculateFreeSpaceLoss(dist))
        if (receivedPower >= self.SENSITIVITY):
            return self.calculateIdealDataRate(dist) * (1 - self.calculateBER(dist))
        else:
            return 0
    
    '''
        PARAMETER UPDATE FUNCTIONS
    '''
    def DataSent(self, dist,dt):
        return self.calculateEffectiveDataRate(dist) * dt
    
    def DataToSend(self, data_sent):
        self.current_data = self.current_data - data_sent
        if self.current_data < 0:
            self.current_data = 0
    
    #def reduceAvailableEnergy(self, dt):
    #    self.AVAILABLEENERGY = self.AVAILABLEENERGY - self.POWER * dt
    
    '''
        RUN FUNCTION - CALL THIS TO RUN THE DATA CLASS
    '''    
    #def run(self, dist, dt):
     #   self.reduceDataRate(dist, dt)
      #  self.reduceAvailableEnergy(dt)
       # return [self.DATATOSEND, self.AVAILABLEENERGY]