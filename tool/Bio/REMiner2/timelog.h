#ifndef	_TIME_H_
#define	_TIME_H_



#define MAX_TIME_STAGE		10		// Not yet (Please...)
#define MAX_TIME_LOG		2048	// Check it (Entire time log number)


int TimeLogPush(char* pbLogMsg, int nStage, int nIsStart);
int TimeLogPop(UINT4 uQryLen, UINT4 uDataLen);
int TimeProgress(char* pbStateName, long long llCurState, long long llLastState, UINT4 uCycle);



#endif
