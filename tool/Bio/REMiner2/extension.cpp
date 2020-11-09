#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <omp.h>
#include "GVD.H"
#include "structure.h"
#include "extension.h"
#include "timelog.h"

int UngappedExt(PTMP_SEED pstTmpSeed, char* pbSeq1, char* pbSeq2, UINT4 uQryLen, UINT4 uDataLen)
{
	UINT4	i	= 0;

	UINT4	uExtNum		= 0;

	UINT4	uMaxBound	= 0;
	UINT4	uMinBound	= 0;

	UINT4	uIndex1		= 0;
	UINT4	uIndex2		= 0;

	int		nMaxScore	= 0;
	int		nScore		= 0;

	
	/* forward extension */
	uExtNum		= 0;
	nMaxScore	= 0;
	nScore		= 0;
	uMaxBound	= MIN2( uQryLen - pstTmpSeed->uX, uDataLen - pstTmpSeed->uY) - WORD_SIZE;

	for ( i = 0 ; i < uMaxBound ; ++i )
	{
		uIndex1		= pstTmpSeed->uX + pstTmpSeed->uLen + i;
		uIndex2		= pstTmpSeed->uY + pstTmpSeed->uLen + i;

		if ( auBase2NumMap[ pbSeq1[uIndex1] - ASCII_A ] > 3 )			// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else if ( auBase2NumMap[ pbSeq2[uIndex2] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else
		{
			if ( pbSeq1[uIndex1] == pbSeq2[uIndex2] )
			{
				nScore	+= (int)SCORE_MAT;

				if ( nScore > nMaxScore )
				{
					nMaxScore	= nScore;

					uExtNum		= i + 1;
				}
			}
			else
			{
				nScore	+= (int)SCORE_MIS;

				if ( nScore < (int)SCORE_THR )
				{
					break;
				}
			}
		}
	}

	pstTmpSeed->uLen	+= uExtNum;


	/* backward extension */
	uExtNum		= 0;
	nMaxScore	= 0;
	nScore		= 0;
	uMinBound	= MIN2(pstTmpSeed->uX,pstTmpSeed->uY);

	for ( i = 0 ; i < uMinBound ; ++i )
	{
		uIndex1		= pstTmpSeed->uX - 1 - i;
		uIndex2		= pstTmpSeed->uY - 1 - i;

		if ( auBase2NumMap[ pbSeq1[uIndex1] - ASCII_A ] > 3 )			// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else if ( auBase2NumMap[ pbSeq2[uIndex2] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else
		{
			if ( pbSeq1[uIndex1] == pbSeq2[uIndex2] )
			{
				nScore	+= (int)SCORE_MAT;

				if ( nScore > nMaxScore )
				{
					nMaxScore	= nScore;

					uExtNum		= i + 1;
				}
			}
			else
			{
				nScore	+= (int)SCORE_MIS;

				if ( nScore < (int)SCORE_THR )
				{
					break;
				}
			}
		}
	}

	pstTmpSeed->uX		-= uExtNum;
	pstTmpSeed->uY		-= uExtNum;
	pstTmpSeed->uLen	+= uExtNum;


	return 0;
}

int GappedExtQry(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	int		i	= 0;

	RE		stRE		= {0};
	TMP_SEED	stTmpSeed	= {0};
	long long	llDiag		= 0;
	
	UINT4		uMatNum1	= 0;
	UINT4		uDiff1		= 0;
	UINT4		uMatNum2	= 0;
	UINT4		uDiff2		= 0;

	int		nThreadNum	= 0;


//	omp_set_num_threads(1);

#pragma omp parallel for private (stRE, stTmpSeed, llDiag, uMatNum1, uDiff1, uMatNum2, uDiff2, nThreadNum) schedule (dynamic)
	for ( i = 0 ; i < pstRealSeedList->nSeedNum ; ++i )
	{
		TimeProgress("Extension (Qry)", i, pstRealSeedList->nSeedNum, 40000);

		stTmpSeed		= GetRealSeed(pstRealSeedList, i);

		llDiag			= (long long) (stTmpSeed.uX - stTmpSeed.uY);
		stRE.uX1		= stTmpSeed.uX;
		stRE.uX2		= stTmpSeed.uX + stTmpSeed.uLen - 1;
		stRE.uY1		= stTmpSeed.uY;
		stRE.uY2		= stTmpSeed.uY + stTmpSeed.uLen - 1;
		stRE.uSeedX		= stTmpSeed.uX;
		stRE.uSeedY		= stTmpSeed.uY;
		stRE.uSeedLen		= stTmpSeed.uLen;

		nThreadNum		= omp_get_thread_num();

		GappedExtForward(pbQry, pbData, uQryLen, uDataLen, &stRE, &uMatNum1, &uDiff1, nThreadNum);
		GappedExtBackward(pbQry, pbData, uQryLen, uDataLen, &stRE, &uMatNum2, &uDiff2, nThreadNum);

		stRE.fIdentity	= (float) (stTmpSeed.uMatNum + uMatNum1 + uMatNum2) / (stTmpSeed.uLen + (uMatNum1 + uDiff1) + (uMatNum2 + uDiff2));

		WriteResFileQry(&stRE, uQryLen, uDataLen);
	}


	return 0;
}

int GappedExtRC(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	int		i	= 0;

	RE		stRE		= {0};
	TMP_SEED	stTmpSeed	= {0};

	UINT4		uMatNum1	= 0;
	UINT4		uDiff1		= 0;
	UINT4		uMatNum2	= 0;
	UINT4		uDiff2		= 0;

	int		nThreadNum	= 0;


//	omp_set_num_threads(1);

#pragma omp parallel for private (stRE, stTmpSeed, uMatNum1, uDiff1, uMatNum2, uDiff2, nThreadNum) schedule (dynamic)
	for ( i = 0 ; i < pstRealSeedList->nSeedNum ; ++i )
	{
		TimeProgress("Extension (RC)", i, pstRealSeedList->nSeedNum, 40000);

		stTmpSeed		= GetRealSeed(pstRealSeedList, i);

		stRE.uX1		= stTmpSeed.uX;
		stRE.uX2		= stTmpSeed.uX + stTmpSeed.uLen - 1;
		stRE.uY1		= stTmpSeed.uY;
		stRE.uY2		= stTmpSeed.uY + stTmpSeed.uLen - 1;
		stRE.uSeedX		= stTmpSeed.uX;
		stRE.uSeedY		= stTmpSeed.uY;
		stRE.uSeedLen		= stTmpSeed.uLen;

		nThreadNum		= omp_get_thread_num();

		GappedExtForward(pbQry, pbRC, uQryLen, uDataLen, &stRE, &uMatNum1, &uDiff1, nThreadNum);
		GappedExtBackward(pbQry, pbRC, uQryLen, uDataLen, &stRE, &uMatNum2, &uDiff2, nThreadNum);

		stRE.fIdentity	= (float) (stTmpSeed.uMatNum + uMatNum1 + uMatNum2) / (stTmpSeed.uLen + (uMatNum1 + uDiff1) + (uMatNum2 + uDiff2));

		WriteResFileRC(&stRE, uQryLen, uDataLen);
	}


	return 0;
}

int GappedExtForward(char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen, PRE pstRE, UINT4* puMatNum, UINT4* puDiff, int nThreadNum)
{
	UINT4	uX		= 0;
	UINT4	uY		= 0;
	UINT4	uMaxI	= 0;
	UINT4	uMaxJ	= 0;

	long long	llGreedyI			= 0;			// i
	long long	llGreedyJ			= 0;			// j
	long long	llGreedyK			= 0;			// k
	float		fGreedyTp			= (float)0;		// T'
	float*		pfGreedyT			= NULL;			// T[d]
	float		fGreedySp			= (float)0;		// S'
	long long	llGreedyD			= 0;			// d
	long long	llGreedyDp			= 0;			// d'
	long long	llGreedyPreL		= 0;			// L (previous)
	long long	llGreedyPreU		= 0;			// U (previous)
	long long	llGreedyL			= 0;			// L
	long long	llGreedyU			= 0;			// U
	long long	llGreedyMinL		= 0;			// min L (line 19)
	long long	llGreedyMaxU		= 0;			// max U (line 20)
	long long	llGreedyM			= 0;			// M
	long long	llGreedyN			= 0;			// N
	float		fGreedyInd			= (float)0;		// ind
	long long	llGreedyFloor		= 0;			// floor value (line 8)

	UINT4	uRStartIndex	= 0;
	UINT4	uRLength		= 0;

	long long	llMinL		= (long long)(INFINITY8);	// minimum value of L
	long long	llMaxU		= (long long)(-INFINITY8);	// maximum value of U
	
	int		nIsNoBase	= 0;


	fGreedyInd		= (float)SCORE_MIS - (float)SCORE_MAT / 2;
	llGreedyFloor	= (long long) floor ( (float) ( GREEDY_X + (float)SCORE_MAT / 2 ) / ( SCORE_MAT - SCORE_MIS ) );
	pfGreedyT		= (float*) calloc ( (size_t)llGreedyFloor + 2, sizeof(float) );

	uX			= pstRE->uX2;		// start x
	uY			= pstRE->uY2;		// start y
	
	llGreedyM	= (long long) ( uQryLen - uX - 1 );
	llGreedyN	= (long long) ( uDataLen - uY - 1 );


	while ( llGreedyL <= llGreedyU + 2 )
	{
		llGreedyD	= llGreedyD + 1;
		llGreedyDp	= llGreedyD - llGreedyFloor - 1;

		uRStartIndex	= (UINT4) ( llGreedyPreL - 1 + (-GREEDY_MIN_L+2) );
		uRLength		= (UINT4) ( llGreedyPreU - llGreedyPreL + 3 );

		memcpy(g_ppuPreMatNumArr[nThreadNum] + uRStartIndex, g_ppuCurMatNumArr[nThreadNum] + uRStartIndex, uRLength * sizeof(UINT4));

		memcpy(g_ppllGreedyPreR[nThreadNum] + uRStartIndex, g_ppllGreedyCurR[nThreadNum] + uRStartIndex, uRLength * sizeof(long long));
		memset(g_ppllGreedyCurR[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(long long));

		memcpy(pfGreedyT, pfGreedyT + 1, (size_t) (llGreedyFloor + 1) * sizeof(float));

		llGreedyPreL	= llGreedyL;
		llGreedyPreU	= llGreedyU;

		llGreedyMinL	= (long long)(INFINITY8);
		llGreedyMaxU	= (long long)(-INFINITY8);

		if ( llGreedyL < llMinL )
		{
			llMinL		= llGreedyL;
		}
		if ( llGreedyU > llMaxU )
		{
			llMaxU		= llGreedyU;
		}


		for ( llGreedyK = llGreedyL - 1 ; llGreedyK <= llGreedyU + 1 ; ++llGreedyK )
		{
			llGreedyI		= GetGreedyMaxI(llGreedyK, llGreedyL, llGreedyU, g_ppllGreedyPreR[nThreadNum], pbQry + uX, pbRC + uY, g_ppuPreMatNumArr[nThreadNum], g_ppuCurMatNumArr[nThreadNum]);
			llGreedyJ		= llGreedyI - llGreedyK;

			fGreedySp		= GetGreedySp(llGreedyI, llGreedyJ, llGreedyD);

			if ( ( llGreedyI >= 0 ) && ( fGreedySp >= (pfGreedyT[0] - GREEDY_X) ) )		// pfGreedyT[0]: T[d']
			{
				nIsNoBase	= 0;

				while ( (llGreedyI < llGreedyM) && (llGreedyJ < llGreedyN) )
				{
					if ( auBase2NumMap[ pbQry[uX + llGreedyI + 1] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
					{
						nIsNoBase	= 1;
						break;
					}
					else if ( auBase2NumMap[ pbRC[uY + llGreedyJ + 1] - ASCII_A ] > 3 )	// Exist anything except A, C, G, T (no base)
					{
						nIsNoBase	= 1;
						break;
					}
					else if ( pbQry[uX + llGreedyI + 1] == pbRC[uY + llGreedyJ + 1] )
					{
						g_ppuCurMatNumArr[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	+= 1;
						llGreedyI	+= 1;
						llGreedyJ	+= 1;
					}
					else
					{
						break;
					}
				}

				fGreedySp	= GetGreedySp(llGreedyI, llGreedyJ, llGreedyD);

				if ( fGreedyTp < fGreedySp )
				{
					fGreedyTp	= fGreedySp;

					uMaxI		= (UINT4) llGreedyI;
					uMaxJ		= (UINT4) llGreedyJ;

					*puMatNum	= g_ppuCurMatNumArr[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)];
					*puDiff		= (UINT4) llGreedyD;
				}

				if ( (llGreedyI == llGreedyM) || (llGreedyJ == llGreedyN) )		// end of file
				{
					llGreedyMinL	= (long long)(INFINITY8);
					llGreedyMaxU	= (long long)(-INFINITY8);

					break;
				}

				if ( nIsNoBase )
				{
					g_ppllGreedyCurR[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	= (long long)(-INFINITY8);
				}
				else
				{
					g_ppllGreedyCurR[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	= llGreedyI;

					if ( llGreedyK < llGreedyMinL )
					{
						llGreedyMinL	= llGreedyK;
					}
					if ( llGreedyK > llGreedyMaxU )
					{
						llGreedyMaxU	= llGreedyK;
					}
				}
			}
			else
			{
				g_ppllGreedyCurR[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	= (long long)(-INFINITY8);
			}
		}

		pfGreedyT[llGreedyFloor+1]	= fGreedyTp;	// pfGreedyT[llGreedyFloor+1]: T[d]

		llGreedyL	= llGreedyMinL;
		llGreedyU	= llGreedyMaxU;

		if ( llGreedyL < GREEDY_MIN_L )
		{
			printf("Too big GREEDY_L\n");
			
			break;
		}
		if ( llGreedyU > GREEDY_MAX_U )
		{
			printf("Too small GREEDY_U\n");
			
			break;
		}
	}

	uRStartIndex	= (UINT4) ( llGreedyPreL - 1 + (-GREEDY_MIN_L+2) );
	uRLength		= (UINT4) ( llGreedyPreU - llGreedyPreL + 3 );

	memset(g_ppllGreedyCurR[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(long long));

	uRStartIndex	= (UINT4) ( llMinL - 1 + (-GREEDY_MIN_L+2) );
	uRLength		= (UINT4) ( llMaxU - llMinL + 3 );

	memset(g_ppllGreedyPreR[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(long long));
	memset(g_ppuPreMatNumArr[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(UINT4));
	memset(g_ppuCurMatNumArr[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(UINT4));

	
	/* update RE */
	pstRE->uX2		+= uMaxI;
	pstRE->uY2		+= uMaxJ;


	free(pfGreedyT);

	return 0;
}


int GappedExtBackward(char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen, PRE pstRE, UINT4* puMatNum, UINT4* puDiff, int nThreadNum)
{
	UINT4	uX		= 0;
	UINT4	uY		= 0;
	UINT4	uMinI	= 0;
	UINT4	uMinJ	= 0;

	long long	llGreedyI			= 0;			// i
	long long	llGreedyJ			= 0;			// j
	long long	llGreedyK			= 0;			// k
	float		fGreedyTp			= (float)0;		// T'
	float*		pfGreedyT			= NULL;			// T[d]
	float		fGreedySp			= (float)0;		// S'
	long long	llGreedyD			= 0;			// d
	long long	llGreedyDp			= 0;			// d'
	long long	llGreedyPreL		= 0;			// L (previous)
	long long	llGreedyPreU		= 0;			// U (previous)
	long long	llGreedyL			= 0;			// L
	long long	llGreedyU			= 0;			// U
	long long	llGreedyMinL		= 0;			// min L (line 19)
	long long	llGreedyMaxU		= 0;			// max U (line 20)
	long long	llGreedyM			= 0;			// M
	long long	llGreedyN			= 0;			// N
	float		fGreedyInd			= (float)0;		// ind
	long long	llGreedyFloor		= 0;			// floor value (line 8)

	UINT4	uRStartIndex	= 0;
	UINT4	uRLength		= 0;

	long long	llMinL		= (long long)(INFINITY8);	// minimum value of L
	long long	llMaxU		= (long long)(-INFINITY8);	// maximum value of U
	
	int		nIsNoBase	= 0;


	fGreedyInd		= (float)SCORE_MIS - (float)SCORE_MAT / 2;
	llGreedyFloor	= (long long) floor ( (float) ( GREEDY_X + (float)SCORE_MAT / 2 ) / ( SCORE_MAT - SCORE_MIS ) );
	pfGreedyT		= (float*) calloc ( (size_t)llGreedyFloor + 2, sizeof(float) );

	uX			= pstRE->uX1;	// start x
	uY			= pstRE->uY1;	// start y

	llGreedyM	= (long long) uX;
	llGreedyN	= (long long) uY;


	while ( llGreedyL <= llGreedyU + 2 )
	{
		llGreedyD	= llGreedyD + 1;
		llGreedyDp	= llGreedyD - llGreedyFloor - 1;

		uRStartIndex	= (UINT4) ( llGreedyPreL - 1 + (-GREEDY_MIN_L+2) );
		uRLength		= (UINT4) ( llGreedyPreU - llGreedyPreL + 3 );

		memcpy(g_ppuPreMatNumArr[nThreadNum] + uRStartIndex, g_ppuCurMatNumArr[nThreadNum] + uRStartIndex, uRLength * sizeof(UINT4));

		memcpy(g_ppllGreedyPreR[nThreadNum] + uRStartIndex, g_ppllGreedyCurR[nThreadNum] + uRStartIndex, uRLength * sizeof(long long));
		memset(g_ppllGreedyCurR[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(long long));

		memcpy(pfGreedyT, pfGreedyT + 1, (size_t) (llGreedyFloor + 1) * sizeof(float));

		llGreedyPreL	= llGreedyL;
		llGreedyPreU	= llGreedyU;

		llGreedyMinL	= (long long)(INFINITY8);
		llGreedyMaxU	= (long long)(-INFINITY8);

		if ( llGreedyL < llMinL )
		{
			llMinL		= llGreedyL;
		}
		if ( llGreedyU > llMaxU )
		{
			llMaxU		= llGreedyU;
		}


		for ( llGreedyK = llGreedyL - 1 ; llGreedyK <= llGreedyU + 1 ; ++llGreedyK )
		{
			llGreedyI		= GetGreedyMinI(llGreedyK, llGreedyL, llGreedyU, g_ppllGreedyPreR[nThreadNum], pbQry, pbRC, uX, uY, g_ppuPreMatNumArr[nThreadNum], g_ppuCurMatNumArr[nThreadNum]);
			llGreedyJ		= llGreedyI + llGreedyK;

			fGreedySp		= GetGreedySp(-llGreedyI, -llGreedyJ, llGreedyD);

			if ( ( llGreedyI <= 0 ) && ( fGreedySp >= (pfGreedyT[0] - GREEDY_X) ) )		// pfGreedyT[0]: T[d']
			{
				nIsNoBase	= 0;

				while ( ((-llGreedyI) < llGreedyM) && ((-llGreedyJ) < llGreedyN) )
				{
					if ( auBase2NumMap[ pbQry[uX + llGreedyI - 1] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
					{
						nIsNoBase	= 1;
						break;
					}
					else if ( auBase2NumMap[ pbRC[uY + llGreedyJ - 1] - ASCII_A ] > 3 )	// Exist anything except A, C, G, T (no base)
					{
						nIsNoBase	= 1;
						break;
					}
					else if ( pbQry[uX + llGreedyI - 1] == pbRC[uY + llGreedyJ - 1] )
					{
						g_ppuCurMatNumArr[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	+= 1;
						llGreedyI	-= 1;
						llGreedyJ	-= 1;
					}
					else
					{
						break;
					}
				}

				fGreedySp	= GetGreedySp(-llGreedyI, -llGreedyJ, llGreedyD);

				if ( fGreedyTp < fGreedySp )
				{
					fGreedyTp	= fGreedySp;

					uMinI		= (UINT4) (-llGreedyI);
					uMinJ		= (UINT4) (-llGreedyJ);

					*puMatNum	= g_ppuCurMatNumArr[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)];
					*puDiff		= (UINT4) llGreedyD;
				}

				if ( ((-llGreedyI) == llGreedyM) || ((-llGreedyJ) == llGreedyN) )		// end of file
				{
					llGreedyMinL	= (long long)(INFINITY8);
					llGreedyMaxU	= (long long)(-INFINITY8);

					break;
				}

				if ( nIsNoBase )
				{
					g_ppllGreedyCurR[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	= (long long) INFINITY8;
				}
				else
				{
					g_ppllGreedyCurR[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	= llGreedyI;

					if ( llGreedyK < llGreedyMinL )
					{
						llGreedyMinL	= llGreedyK;
					}
					if ( llGreedyK > llGreedyMaxU )
					{
						llGreedyMaxU	= llGreedyK;
					}
				}
			}
			else
			{
				g_ppllGreedyCurR[nThreadNum][llGreedyK + (-GREEDY_MIN_L+2)]	= (long long) INFINITY8;
			}
		}

		pfGreedyT[llGreedyFloor+1]	= fGreedyTp;	// pfGreedyT[llGreedyFloor+1]: T[d]

		llGreedyL	= llGreedyMinL;
		llGreedyU	= llGreedyMaxU;

		if ( llGreedyL < GREEDY_MIN_L )
		{
			printf("Too big GREEDY_L\n");
			
			break;
		}
		if ( llGreedyU > GREEDY_MAX_U )
		{
			printf("Too small GREEDY_U\n");
			
			break;
		}
	}

	uRStartIndex	= (UINT4) ( llGreedyPreL - 1 + (-GREEDY_MIN_L+2) );
	uRLength		= (UINT4) ( llGreedyPreU - llGreedyPreL + 3 );

	memset(g_ppllGreedyCurR[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(long long));

	uRStartIndex	= (UINT4) ( llMinL - 1 + (-GREEDY_MIN_L+2) );
	uRLength		= (UINT4) ( llMaxU - llMinL + 3 );

	memset(g_ppllGreedyPreR[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(long long));
	memset(g_ppuPreMatNumArr[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(UINT4));
	memset(g_ppuCurMatNumArr[nThreadNum] + uRStartIndex, 0, uRLength * sizeof(UINT4));

		
	/* update RE */
	pstRE->uX1		-= uMinI;
	pstRE->uY1		-= uMinJ;


	free(pfGreedyT);

	return 0;
}

float GetGreedySp(long long llI, long long llJ, long long llD)
{
	float fSp	= (float)0;

	fSp	= (float) (llI + llJ) * SCORE_MAT / 2 - (float) llD * (SCORE_MAT - SCORE_MIS) ;

	return fSp;
}

long long GetGreedyMaxI(long long llGreedyK, long long llGreedyL, long long llGreedyU, long long* pllGreedyPreR, char* pbSeq1, char* pbSeq2, UINT4* puPreMatNumArr, UINT4* puCurMatNumArr)
{
	long long	llMaxI	= (long long) (-INFINITY8);
	long long	llTmpI1	= (long long) (-INFINITY8);
	long long	llTmpI2	= (long long) (-INFINITY8);
	long long	llTmpI3	= (long long) (-INFINITY8);


	if ( llGreedyL < llGreedyK )
	{
		llTmpI1	= pllGreedyPreR[llGreedyK - 1 + (-GREEDY_MIN_L+2)] + 1;

		if ( llTmpI1 >= 0)
		{
			if ( auBase2NumMap[ pbSeq1[llTmpI1] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
			{
				llTmpI1		= (long long) (-INFINITY8);
			}
		}
	}

	if ( ( llGreedyL <= llGreedyK ) && ( llGreedyK <= llGreedyU ) )
	{
		llTmpI2	= pllGreedyPreR[llGreedyK + (-GREEDY_MIN_L+2)] + 1;

		if ( llTmpI2 >= 0)
		{
			if ( ( auBase2NumMap[ pbSeq1[llTmpI2] - ASCII_A ] > 3 ) && ( auBase2NumMap[ pbSeq2[llTmpI2] - ASCII_A ] > 3 ) )		// Exist anything except A, C, G, T (no base)
			{
				llTmpI2		= (long long) (-INFINITY8);
			}
		}
	}

	if ( llGreedyK < llGreedyU )
	{
		llTmpI3	= pllGreedyPreR[llGreedyK + 1 + (-GREEDY_MIN_L+2)];

		if ( llTmpI3 >= 0)
		{
			if ( auBase2NumMap[ pbSeq2[llTmpI3] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
			{
				llTmpI3		= (long long) (-INFINITY8);
			}
		}
	}

	if ( llTmpI1 > llTmpI2 )
	{
		if ( llTmpI3 > llTmpI1 )	// tmp3
		{
			llMaxI	= llTmpI3;

			puCurMatNumArr[llGreedyK + (-GREEDY_MIN_L+2)]	= puPreMatNumArr[llGreedyK + 1 + (-GREEDY_MIN_L+2)];
		}
		else						// tmp1
		{
			llMaxI	= llTmpI1;

			puCurMatNumArr[llGreedyK + (-GREEDY_MIN_L+2)]	= puPreMatNumArr[llGreedyK - 1 + (-GREEDY_MIN_L+2)];
		}
	}
	else
	{
		if ( llTmpI3 > llTmpI2 )	// tmp3
		{
			llMaxI	= llTmpI3;

			puCurMatNumArr[llGreedyK + (-GREEDY_MIN_L+2)]	= puPreMatNumArr[llGreedyK + 1 + (-GREEDY_MIN_L+2)];
		}
		else						// tmp2
		{
			llMaxI	= llTmpI2;
		}
	}


	return llMaxI;
}

long long GetGreedyMinI(long long llGreedyK, long long llGreedyL, long long llGreedyU, long long* pllGreedyPreR, char* pbSeq1, char* pbSeq2, UINT4 uX, UINT4 uY, UINT4* puPreMatNumArr, UINT4* puCurMatNumArr)
{
	long long	llMinI	= (long long) INFINITY8;
	long long	llTmpI1	= (long long) INFINITY8;
	long long	llTmpI2	= (long long) INFINITY8;
	long long	llTmpI3	= (long long) INFINITY8;


	if ( llGreedyL < llGreedyK )
	{
		llTmpI1	= pllGreedyPreR[llGreedyK - 1 + (-GREEDY_MIN_L+2)] - 1;

		if ( llTmpI1 <= 0)
		{
			if ( auBase2NumMap[ pbSeq1[(long long)uX + llTmpI1] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
			{
				llTmpI1		= (long long) INFINITY8;
			}
		}
	}

	if ( ( llGreedyL <= llGreedyK ) && ( llGreedyK <= llGreedyU ) )
	{
		llTmpI2	= pllGreedyPreR[llGreedyK + (-GREEDY_MIN_L+2)] - 1;

		if ( llTmpI2 <= 0)
		{
			if ( ( auBase2NumMap[ pbSeq1[(long long)uX + llTmpI2] - ASCII_A ] > 3 ) && ( auBase2NumMap[ pbSeq2[(long long)uY + llTmpI2] - ASCII_A ] > 3 ) )		// Exist anything except A, C, G, T (no base)
			{
				llTmpI2		= (long long) INFINITY8;
			}
		}
	}

	if ( llGreedyK < llGreedyU )
	{
		llTmpI3	= pllGreedyPreR[llGreedyK + 1 + (-GREEDY_MIN_L+2)];

		if ( llTmpI3 <= 0)
		{
			if ( auBase2NumMap[ pbSeq2[(long long)uY + llTmpI3] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
			{
				llTmpI3		= (long long) INFINITY8;
			}
		}
	}

	if ( llTmpI1 < llTmpI2 )
	{
		if ( llTmpI3 < llTmpI1 )	// tmp3
		{
			llMinI	= llTmpI3;

			puCurMatNumArr[llGreedyK + (-GREEDY_MIN_L+2)]	= puPreMatNumArr[llGreedyK + 1 + (-GREEDY_MIN_L+2)];
		}
		else						// tmp1
		{
			llMinI	= llTmpI1;

			puCurMatNumArr[llGreedyK + (-GREEDY_MIN_L+2)]	= puPreMatNumArr[llGreedyK - 1 + (-GREEDY_MIN_L+2)];
		}
	}
	else
	{
		if ( llTmpI3 < llTmpI2 )	// tmp3
		{
			llMinI	= llTmpI3;

			puCurMatNumArr[llGreedyK + (-GREEDY_MIN_L+2)]	= puPreMatNumArr[llGreedyK + 1 + (-GREEDY_MIN_L+2)];
		}
		else						// tmp2
		{
			llMinI	= llTmpI2;
		}
	}


	return llMinI;
}

int WriteResFileQry(PRE pstRE, UINT4 uQryLen, UINT4 uDataLen)
{
#pragma omp critical (file_lock)
	{
		fwrite(pstRE, sizeof(RE), 1, g_pfRes);
	}

	return 0;
}

int WriteResFileRC(PRE pstRE, UINT4 uQryLen, UINT4 uDataLen)
{
	pstRE->uY1	= uQryLen - pstRE->uY1 - 1;
	pstRE->uY2	= uDataLen - pstRE->uY2 - 1;

#pragma omp critical (file_lock)
	{
		fwrite(pstRE, sizeof(RE), 1, g_pfRes);
	}

	return 0;
}
