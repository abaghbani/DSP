
class Constant:
	
	DpskSync = [3,-3,3,-3,3,-3,-3,3,3,3]
	DpskSyncHdr4 = [-1,1,1,3,-1,-3,-1,3,3,1,1,1,-3,-1,1,-1,1,-3,1,-3,-3,3,-3,-3,-3,1,-3,1,-3,-3,1,1,-3,1,1,3,-1,-3,-3,-1,1,-3,1,-1,3,-1,1,1,1,1,1,1,3,3,1,1,3,3,1]
	DpskSyncHdr8 = [-1,3,-1,1,3,-3,-3,-1,-1,1,-3,-3,1,-3,1,1,-1,-1,3,-3,3,-3,3,-3,-3,3,-3,3,-1,-1,1,1,3,-1,-1,3,1,-1,3,3,-1,-3,-1,-1,-1,-3,-3,3,3,-1,-1,-3,3,-1,-3,-1,1,-3,1,3,3,1,3,-3,-1,-1,1,3,3,1,3,3,-3,-3,-3,1,3,-1,3,1,1,1,-1,1,3,-3,1,-1,3,1,-3,-3,1,3,3,1,-3,1,-1,3,-1,-1,-1,1,1,-3,-3,3,3,3,-1,-1,-1,-3,1,1,3,-1,1]
	
	class DpskModulationType:
		Edr2 = 1
		Edr3 = 2

	## Dpsk demodulator
	TableRxPhase4DQPSK = [100, 1, 1, 100, 100, 3, 3, 100, 100, -3, -3, 100, 100, -1, -1, 100]
	TableRxPhase8DQPSK = [0, 1, 1, 2, 2, 3, 3, 4, 4, -3, -3, -2, -2, -1, -1, 0]

