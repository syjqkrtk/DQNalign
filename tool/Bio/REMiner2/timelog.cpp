#include <stdio.h>
#include <time.h>
#include <string.h>
#include "GVD.H"
#include "timelog.h"


char	g_abTimeLogMsg[MAX_TIME_LOG] = {0};


int TimeLogPush(char* pbLogMsg, int nStage, int nIsStart)
{
	int i = 0;

	time_t		tCurTime	= 0;		// Current time
	struct tm*	tmCurTime	= {0};		// Current time

	static time_t	s_tStartTime[MAX_TIME_STAGE]	= {0};	// Start time
	time_t			tElapsedTime					= 0;	// Time difference

	char abTimeLogMsg[MAX_TIME_LOG]		= {0};
	char abElapsedTime[MAX_TIME_LOG]	= {0};
	char abTap[MAX_TIME_STAGE*2]		= {0};		// About stage
	char abStart[8]						= {0};		// About start, end

	int	errno;


	/* Current time */
	tCurTime	= time(NULL);							// Current time (From 1970Y 1M 1d 0h 0m 0s)
	tmCurTime		= localtime(&tCurTime);	// Save current time into tm structure


	/* Record time */
	if ( nIsStart == 1)
	{
		s_tStartTime[nStage]	= tCurTime;
	}
	else
	{
		tElapsedTime	= tCurTime - s_tStartTime[nStage];
	}


	/* Empty/start/end char */
	for (i = 0; i < nStage; ++i)
	{
		sprintf(abTap + i*2, "  ");
	}

	if ( nIsStart == 1)
	{
		strcpy(abStart, "START");
	}
	else
	{
		strcpy(abStart, " END ");
	}
	

//	sprintf(abTimeLogMsg,
//		"%s%s (%s) : %d%d%d, %dH, %dM, %dS\n",
//		abTap, pbLogMsg, abStart,
//		tmCurTime->tm_mon + 1, tmCurTime->tm_mday, tmCurTime->tm_year + 1900, tmCurTime->tm_hour, tmCurTime->tm_min, tmCurTime->tm_sec);	// Years

	sprintf ( abTimeLogMsg,
			"%s%s (%s) : %d/%d, %d:%d:%d",
			abTap, pbLogMsg, abStart,
			tmCurTime->tm_mon + 1,	tmCurTime->tm_mday,	tmCurTime->tm_hour,	tmCurTime->tm_min,	tmCurTime->tm_sec );

	if ( nIsStart == 0)
	{
		sprintf ( abElapsedTime, " (%d)", tElapsedTime );

		strcat ( abTimeLogMsg, abElapsedTime );
	}

	strcat ( abTimeLogMsg, "\n");

	//printf("%s\n", abTimeLogMsg);


	if ( strlen(g_abTimeLogMsg) + strlen(abTimeLogMsg) > MAX_TIME_LOG )
	{
		//printf("No space to write time log\n");
		return -1;
	}
	else
	{
		strcat( g_abTimeLogMsg, abTimeLogMsg );
	}

	return 0;
}

int TimeLogPop(UINT4 uQryLen, UINT4 uDataLen)
{
	/* (1) Print to file */
	FILE*	pfLog	= NULL;

	int	errno;

	pfLog = fopen(LOG_FILE, "w");

	fprintf(pfLog, "\n\n************************************************************************");
	fprintf(pfLog, "\n%s\n", g_abTimeLogMsg);
	fprintf(pfLog, "Input File Name: %s\n", INPUT_FILE1);
	fprintf(pfLog, "Input Data Length: %u Bytes\n", uQryLen);
	fprintf(pfLog, "Input File Name: %s\n", INPUT_FILE2);
	fprintf(pfLog, "Input Data Length: %u Bytes\n", uDataLen);
	fprintf(pfLog, "W: %u, m: %d, SP: %u, L: %u\n", WORD_SIZE, ALLOW_SIZE, SPACE_SIZE, MIN_SEED_LEN);
	fprintf(pfLog, "SCORE: Match (%d), Mismatch (%d), Threshold (%d), X (%d)\n", (int)SCORE_MAT, (int)SCORE_MIS, (int)SCORE_THR, (int)GREEDY_X);
	fprintf(pfLog, "Filtering: Window Size (%d), Score Threshold (%.2f)\n", (int)WD_SIZE, (float)T_THR);
	fprintf(pfLog, "Time Log Length: %d Bytes (MAX: %d Bytes)\n", strlen(g_abTimeLogMsg), (int)MAX_TIME_LOG);
	fprintf(pfLog, "************************************************************************\n\n");
	fprintf(pfLog, "\n");

	fclose(pfLog);



	/* (2) Print to screen */
	/*
	printf("\n\n************************************************************************");
	printf("\n%s\n", g_abTimeLogMsg);
	printf("Input File Name: %s\n", INPUT_FILE1);
	printf("Input Data Length: %u Bytes\n", uQryLen);
	printf("Input File Name: %s\n", INPUT_FILE2);
	printf("Input Data Length: %u Bytes\n", uDataLen);
	printf("W: %u, m: %d, SP: %u, L: %u\n", WORD_SIZE, ALLOW_SIZE, SPACE_SIZE, MIN_SEED_LEN);
	printf("SCORE: Match (%d), Mismatch (%d), Threshold (%d), X (%d)\n", (int)SCORE_MAT, (int)SCORE_MIS, (int)SCORE_THR, (int)GREEDY_X);
	printf("Filtering: Window Size (%d), Score Threshold (%.2f)\n", (int)WD_SIZE, (float)T_THR);
	printf("Time Log Length: %d Bytes (MAX: %d Bytes)\n", strlen(g_abTimeLogMsg), (int)MAX_TIME_LOG);
	printf("************************************************************************\n\n");
	printf("\n");
	*/



	return 0;
}

int TimeProgress(char* pbStateName, long long llCurState, long long llLastState, UINT4 uCycle)
{
	if ( ( llCurState % uCycle ) == 0 )
	{
		//printf("Time Progress: [ %s ] %lld / %lld\n", pbStateName, llCurState, llLastState);
	}

	return 0;
}
