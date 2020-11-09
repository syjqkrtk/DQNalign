#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include "GVD.H"
#include "structure.h"



int InitTmpSeedList(PTMP_SEED_LIST pstTmpSeedList, long long llDiagNum)
{
	pstTmpSeedList->llDiagNum	= llDiagNum;

	pstTmpSeedList->pTmpSeed	= (PTMP_SEED) calloc ( (size_t)llDiagNum, sizeof(TMP_SEED) );

	return 0;
}

int UpdateTmpSeed(PTMP_SEED_LIST pstTmpSeedList, long long llDiag, UINT4 uX, UINT4 uY, UINT4 uLen, UINT4 uMatNum)
{
	pstTmpSeedList->pTmpSeed[llDiag].uX			= uX;
	pstTmpSeedList->pTmpSeed[llDiag].uY			= uY;
	pstTmpSeedList->pTmpSeed[llDiag].uLen		= uLen;
	pstTmpSeedList->pTmpSeed[llDiag].uMatNum	= uMatNum;

	return 0;
}

int UpdateTmpSeed(PTMP_SEED_LIST pstTmpSeedList, long long llDiag, UINT4 uX, UINT4 uY, UINT4 uLen)
{
	pstTmpSeedList->pTmpSeed[llDiag].uX			= uX;
	pstTmpSeedList->pTmpSeed[llDiag].uY			= uY;
	pstTmpSeedList->pTmpSeed[llDiag].uLen		= uLen;

	return 0;
}

int FreeTmpSeedList(PTMP_SEED_LIST pstTmpSeedList)
{
	free ( pstTmpSeedList->pTmpSeed );
	pstTmpSeedList->pTmpSeed	= NULL;
	pstTmpSeedList->llDiagNum	= 0;

	return 0;
}

int InitRealSeedList(PREAL_SEED_LIST pstRealSeedList, int nSeedNum)
{
	pstRealSeedList->nSeedNum	= 0;

	pstRealSeedList->pTmpSeed	= (PTMP_SEED) calloc ( (size_t)REAL_SEED_NUM, sizeof(TMP_SEED) );

	return 0;
}

int PushRealSeed(PREAL_SEED_LIST pstRealSeedList, PTMP_SEED pstTmpSeed)
{
	int	nSeedNum	= pstRealSeedList->nSeedNum;

	pstRealSeedList->pTmpSeed[nSeedNum].uX		= pstTmpSeed->uX;
	pstRealSeedList->pTmpSeed[nSeedNum].uY		= pstTmpSeed->uY;
	pstRealSeedList->pTmpSeed[nSeedNum].uLen	= pstTmpSeed->uLen;
	pstRealSeedList->pTmpSeed[nSeedNum].uMatNum	= pstTmpSeed->uMatNum;

	pstRealSeedList->nSeedNum	= pstRealSeedList->nSeedNum	+ 1;

	return 0;
}

TMP_SEED GetRealSeed(PREAL_SEED_LIST pstRealSeedList, int nSeedIndex)
{
	TMP_SEED	stTmpSeed	= pstRealSeedList->pTmpSeed[nSeedIndex];

	return stTmpSeed;
}

int ResetRealSeedList(PREAL_SEED_LIST pstRealSeedList)
{
	pstRealSeedList->nSeedNum	= 0;

	return 0;
}

int FreeRealSeedList(PREAL_SEED_LIST pstRealSeedList)
{
	free ( pstRealSeedList->pTmpSeed );
	pstRealSeedList->pTmpSeed	= NULL;
	pstRealSeedList->nSeedNum	= 0;

	return 0;
}
