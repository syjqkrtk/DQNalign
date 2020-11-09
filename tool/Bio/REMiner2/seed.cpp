#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "GVD.H"
#include "structure.h"
#include "seed.h"
#include "extension.h"
#include "timelog.h"


int WordMatch(char* pbQryM, UINT4 uQryLen, PWORD_ELE pstWordEle, PTMP_WORD pstTmpWord)
{
	UINT4	i	= 0;

	UINT4	uWord		= 0;
	UINT4	uPrevWord	= 0;
	int	nNoBaseOffset	= 0;


	while ( i < uQryLen - WORD_SIZE + 1 )
	{
		TimeProgress("Word Match", i, uQryLen, 1000000);

		uWord = GetWord ( pbQryM + i, &nNoBaseOffset );

		if ( nNoBaseOffset < 0 )	// Only A,C,G,T
		{
			uPrevWord	= pstTmpWord[uWord].uIndex;

			pstWordEle[i].uIndex		= i;
			pstWordEle[i].uPrevIndex	= uPrevWord;
			pstTmpWord[uWord].uIndex	= i;

			i += 1;
		}
		else
		{
			i += ( nNoBaseOffset + 1 );		// Jump non-A,C,G,T and Scan the bases
		}
	}


	return 0;
}

UINT4 GetWord(char* pbSeq, int* pnNoBaseOffset)
{
	int		i = 0;

	UINT4	uWord			= 0;	// Quantized value of word
	UINT4	uCurEle			= 0;	// Digit of current base
	UINT4	uShift			= 0;	// Shift value of current base
	int		nNoBaseOffset	= -1;	// Index of non-A,C,G,T (nothing -1)


	for ( i = ((int)WORD_SIZE - 1) ; i >= 0  ; --i )
	{
		if ( (pbSeq[i] < ASCII_A) || (pbSeq[i] > ASCII_Z) )		// masking
		{
			nNoBaseOffset	= i;

			break;
		}
		else
		{
			uCurEle		= auBase2NumMap[ pbSeq[i] - ASCII_A ];
			uShift		= 2 * (UINT4)( (int)WORD_SIZE - i - 1 );

			if ( uCurEle > 3 )	// non-A, C, G, T
			{
				nNoBaseOffset	= i;

				break;
			}

			uWord	+= ( uCurEle << uShift );
		}
	}

	*pnNoBaseOffset	= nNoBaseOffset;


	return uWord;
}

UINT4 GetWord(char* pbSeq)
{
	UINT4	i = 0;

	UINT4	uWord			= 0;	// Quantized value of word
	UINT4	uCurEle			= 0;	// Digit of current base
	UINT4	uShift			= 0;	// Shift value of current base


	for ( i = 0 ; i < WORD_SIZE  ; ++i )
	{
		uCurEle		= auBase2NumMap[ pbSeq[i] - ASCII_A ];
		uShift		= 2 * ( WORD_SIZE - i - 1 );

		uWord		+= ( uCurEle << uShift );
	}


	return uWord;
}

UINT4 GetRCWord(char* pbSeq)
{
	UINT4	i = 0;

	UINT4	uWord			= 0;	// Quantized value of word
	UINT4	uCurEle			= 0;	// Digit of current base
	UINT4	uShift			= 0;	// Shift value of current base


	for ( i = 0 ; i < WORD_SIZE  ; ++i )
	{
		uCurEle		= auBase2CNumMap[ pbSeq[i] - ASCII_A ];
		uShift		= 2 * i;

		uWord		+= ( uCurEle << uShift );
	}


	return uWord;
}

int NHitMethodQry(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	long long       i       = 0;

	UINT4           uX              = 0;
	UINT4           uY              = 0;
	UINT4           uWord   = 0;

	TMP_SEED_LIST   stTmpSeedList   = {0};
	REAL_SEED_LIST  stRealSeedList  = {0};

	UINT4   SeedNum = 0;

	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);

	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (Qry, 1/2)", uQryLen - i, uQryLen, 500000);

		uX      = pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
                {
			uWord   = GetWord(pbQry + uX);        // reverse complement word

			uY	= pstTmpWord2[uWord].uIndex;

			while ( uY != (UINT4) INFINITY4U )
			{
				SeedMakeQry(&stTmpSeedList, &stRealSeedList, uX, uY, pbQry, pbData, uQryLen, uDataLen);

				uY	= pstWordEle2[uY].uPrevIndex;
			}
		}
	}

	CleanTmpSeedListQry(&stTmpSeedList, &stRealSeedList, pbQry, pbData, uQryLen, uDataLen);		// Empty seed in TmpSeedList

	SeedNum = stRealSeedList.nSeedNum;

	CleanRealSeedListQry(&stRealSeedList, pbQry, pbData, uQryLen, uDataLen);			// Empty seed in RealSeedList

	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int NHitMethodQryAllow(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	UINT4		j	= 0;

	UINT4		uX		= 0;
	UINT4		uY		= 0;
	UINT4		uWord1	= 0;
	UINT4		uWord2	= 0;

	TMP_SEED_LIST	stTmpSeedList	= {0};
	REAL_SEED_LIST	stRealSeedList	= {0};
	
	UINT4	uAllowSeqNum	= 0;
	UINT4	SeedNum	= 0;


	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);

	uAllowSeqNum	= GetAllowSeqNum(ALLOW_SIZE);

	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (Qry, 1/2)", uQryLen - i, uQryLen, 500000);

		uX	= pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
		{
			uWord1	= GetWord(pbQry + uX);

			for ( j = 0 ; j < uAllowSeqNum ; ++j )
			{
				uWord2	= GetAllowSeq(uWord1, j);		// If word1 and word2 are same or similar

				uY	= pstTmpWord2[uWord2].uIndex;

				while ( uY != (UINT4) INFINITY4U )
				{
					SeedMakeQry(&stTmpSeedList, &stRealSeedList, uX, uY, pbQry, pbData, uQryLen, uDataLen);

					uY	= pstWordEle2[uY].uPrevIndex;
				}
			}
		}
	}
	CleanTmpSeedListQry(&stTmpSeedList, &stRealSeedList, pbQry, pbData, uQryLen, uDataLen);		// Empty seed in TmpSeedList

	SeedNum = stRealSeedList.nSeedNum;

	CleanRealSeedListQry(&stRealSeedList, pbQry, pbData, uQryLen, uDataLen);			// Empty seed in RealSeedList

	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int NHitMethodRC(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;

	UINT4		uX		= 0;
	UINT4		uY		= 0;
	UINT4		uY2		= 0;
	UINT4		uY2p	= 0;
	UINT4		uWord	= 0;

	TMP_SEED_LIST	stTmpSeedList	= {0};
	REAL_SEED_LIST	stRealSeedList	= {0};

	UINT4	SeedNum	= 0;

	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);


	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (RC, 1/2)", uQryLen - i, uQryLen, 500000);

		uX	= pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
		{
			uWord	= GetRCWord(pbQry + uX);	// reverse complement word

			uY	= pstTmpWord2[uWord].uIndex;

			while ( uY != (UINT4) INFINITY4U )
			{
				uY2	= uY + WORD_SIZE - 1;
				uY2p	= uDataLen - 1 - uY2;

				SeedMakeRC(&stTmpSeedList, &stRealSeedList, uX, uY2p, pbQry, pbRC, uQryLen, uDataLen);

				uY	= pstWordEle2[uY].uPrevIndex;
			}
		}
	}

	CleanTmpSeedListRC(&stTmpSeedList, &stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);		// Empty seed in TmpSeedList

	SeedNum = stRealSeedList.nSeedNum;

	CleanRealSeedListRC(&stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);						// Empty seed in RealSeedList

	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int NHitMethodRCAllow(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	UINT4		j	= 0;

	UINT4		uX		= 0;
	UINT4		uY		= 0;
	UINT4		uY2		= 0;
	UINT4		uY2p	= 0;
	UINT4		uWord1	= 0;
	UINT4		uWord2	= 0;

	TMP_SEED_LIST	stTmpSeedList	= {0};
	REAL_SEED_LIST	stRealSeedList	= {0};
	
	UINT4	uAllowSeqNum	= 0;
	UINT4	SeedNum	= 0;


	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);


	uAllowSeqNum	= GetAllowSeqNum(ALLOW_SIZE);


	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (RC, 1/2)", uQryLen - i, uQryLen, 500000);

		uX	= pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
		{
			uWord1	= GetRCWord(pbQry + uX);

			for ( j = 0 ; j < uAllowSeqNum ; ++j )
			{
				uWord2	= GetAllowSeq(uWord1, j);		// If word1 and word2 are same or similar

				uY	= pstTmpWord2[uWord2].uIndex;

				while ( uY != (UINT4) INFINITY4U )
				{
					uY2	= uY + WORD_SIZE - 1;
					uY2p	= uDataLen - 1- uY2;

					SeedMakeRC(&stTmpSeedList, &stRealSeedList, uX, uY2p, pbQry, pbRC, uQryLen, uDataLen);

					uY	= pstWordEle2[uY].uPrevIndex;
				}
			}
		}
	}

	CleanTmpSeedListRC(&stTmpSeedList, &stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);		// Empty seed in TmpSeedList

	SeedNum = stRealSeedList.nSeedNum;

	CleanRealSeedListRC(&stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);						// Empty seed in RealSeedList
	
	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int SeedMakeQry(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	PTMP_SEED	pstTmpSeed	= NULL;

	long long	llDiag		= 0;

	int		nSpace			= 0;
	int		nIsOnlyBase		= 0;
	int		nIsSeedCombined	= 0;


	llDiag		= (long long) (uDataLen - 1 + uX - uY);

	pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[llDiag]);

	if ( pstTmpSeed->uLen )
	{
		nSpace	= pstTmpSeed->uX - ( uX + WORD_SIZE );

		if ( nSpace <= 0 )	// (1) -WORD < SPACE <= 0
		{
			WordCombine(pstTmpSeed, nSpace);
		}
		
		else if ( nSpace <= (int)SPACE_SIZE )	// (2) 0 < SPACE <= SP
		{
			nIsOnlyBase	= IsOnlyBase(pbQry + uX + WORD_SIZE, pbData + uY + WORD_SIZE, nSpace);
			
			if ( nIsOnlyBase == 1 )		// Only A,C,G,T in SPACE
			{
				WordCombine(pstTmpSeed, nSpace);
			}

			else
			{
				if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Satisfy the threshold of seed
				{
					UngappedExt(pstTmpSeed, pbQry, pbData, uQryLen, uDataLen);

					GappedExtMgrQry(pstRealSeedList, pstTmpSeed, pbQry, pbData, uQryLen, uDataLen);
				}

				UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
			}
		}
		
		else		// (3) SPACE > SP
		{
			if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Satisfy the threshold of seed
			{
				nIsSeedCombined	= SeedCombine(pstTmpSeedList, uX, uY, llDiag, pbQry, pbData);

				if ( ! nIsSeedCombined )
				{
					UngappedExt(pstTmpSeed, pbQry, pbData, uQryLen, uDataLen);

					GappedExtMgrQry(pstRealSeedList, pstTmpSeed, pbQry, pbData, uQryLen, uDataLen);

					UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
				}
			}
			else
			{
				UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
			}
		}
	}

	else	// no element
	{
		UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
	}


	return 0;
}

int SeedMakeRC(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	PTMP_SEED	pstTmpSeed	= NULL;

	long long	llDiag		= 0;

	int		nSpace			= 0;
	int		nIsOnlyBase		= 0;
	int		nIsSeedCombined	= 0;


	llDiag		= (long long) (uDataLen - 1 + uX - uY);

	pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[llDiag]);

	if ( pstTmpSeed->uLen )
	{
		nSpace	= pstTmpSeed->uX - ( uX + WORD_SIZE );

		if ( nSpace <= 0 )	// (1) -WORD < SPACE <= 0
		{
			WordCombine(pstTmpSeed, nSpace);
		}
		
		else if ( nSpace <= (int)SPACE_SIZE )	// (2) 0 < SPACE <= SP
		{
			nIsOnlyBase	= IsOnlyBase(pbQry + uX + WORD_SIZE, pbRC + uY + WORD_SIZE, nSpace);
			
			if ( nIsOnlyBase == 1 )		// Only A,C,G,T in SPACE
			{
				WordCombine(pstTmpSeed, nSpace);
			}

			else
			{
				if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Satisfy the threshold of seed
				{
					UngappedExt(pstTmpSeed, pbQry, pbRC, uQryLen, uDataLen);

					GappedExtMgrRC(pstRealSeedList, pstTmpSeed, pbQry, pbRC, uQryLen, uDataLen);
				}

				UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
			}
		}
		
		else		// (3) SPACE > SP
		{
			if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Satisfy the threshold of seed
			{
				nIsSeedCombined	= SeedCombine(pstTmpSeedList, uX, uY, llDiag, pbQry, pbRC);

				if ( ! nIsSeedCombined )
				{
					UngappedExt(pstTmpSeed, pbQry, pbRC, uQryLen, uDataLen);

					GappedExtMgrRC(pstRealSeedList, pstTmpSeed, pbQry, pbRC, uQryLen, uDataLen);

					UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
				}
			}
			else
			{
				UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
			}
		}
	}

	else	// no element
	{
		UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
	}


	return 0;
}

int NHitMethodQry2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	long long       i       = 0;

	UINT4           uX              = 0;
	UINT4           uY              = 0;
	UINT4           uWord   = 0;

	TMP_SEED_LIST   stTmpSeedList   = {0};
	REAL_SEED_LIST  stRealSeedList  = {0};

	UINT4   SeedNum = 0;

	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);

	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (Qry)", uQryLen - i, uQryLen, 500000);

		uX      = pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
                {
			uWord   = GetWord(pbQry + uX);        // reverse complement word

			uY	= pstTmpWord2[uWord].uIndex;

			while ( uY != (UINT4) INFINITY4U )
			{
				SeedFindQry(&stTmpSeedList, &stRealSeedList, uX, uY, pbQry, pbData, uQryLen, uDataLen);

				uY	= pstWordEle2[uY].uPrevIndex;
			}
		}
	}

	CleanTmpSeedListQry2(&stTmpSeedList, &stRealSeedList, pbQry, pbData, uQryLen, uDataLen);		// Empty seed in TmpSeedList to RealSeedList
	SeedNum = stRealSeedList.nSeedNum;
	CleanRealSeedListQry2(&stRealSeedList, pbQry, pbData, uQryLen, uDataLen);						// print RealSeedList

	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int NHitMethodQryAllow2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	UINT4		j	= 0;

	UINT4		uX		= 0;
	UINT4		uY		= 0;
	UINT4		uWord1	= 0;
	UINT4		uWord2	= 0;

	TMP_SEED_LIST	stTmpSeedList	= {0};
	REAL_SEED_LIST	stRealSeedList	= {0};
	
	UINT4	uAllowSeqNum	= 0;
	UINT4	SeedNum	= 0;


	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);

	uAllowSeqNum	= GetAllowSeqNum(ALLOW_SIZE);

	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (Qry)", uQryLen - i, uQryLen, 500000);

		uX	= pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
		{
			uWord1	= GetWord(pbQry + uX);

			for ( j = 0 ; j < uAllowSeqNum ; ++j )
			{
				uWord2	= GetAllowSeq(uWord1, j);		// If word1 and word2 are same or similar

				uY	= pstTmpWord2[uWord2].uIndex;

				while ( uY != (UINT4) INFINITY4U )
				{
					SeedFindQry(&stTmpSeedList, &stRealSeedList, uX, uY, pbQry, pbData, uQryLen, uDataLen);

					uY	= pstWordEle2[uY].uPrevIndex;
				}
			}
		}
	}

	CleanTmpSeedListQry2(&stTmpSeedList, &stRealSeedList, pbQry, pbData, uQryLen, uDataLen);		// Empty seed in TmpSeedList to RealSeedList
	SeedNum = stRealSeedList.nSeedNum;
	CleanRealSeedListQry2(&stRealSeedList, pbQry, pbData, uQryLen, uDataLen);						// print RealSeedList

	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int NHitMethodRC2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;

	UINT4		uX		= 0;
	UINT4		uY		= 0;
	UINT4		uY2		= 0;
	UINT4		uY2p	= 0;
	UINT4		uWord	= 0;

	TMP_SEED_LIST	stTmpSeedList	= {0};
	REAL_SEED_LIST	stRealSeedList	= {0};

	UINT4	SeedNum	= 0;

	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);


	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (RC)", uQryLen - i, uQryLen, 500000);

		uX	= pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
		{
			uWord	= GetRCWord(pbQry + uX);	// reverse complement word

			uY	= pstTmpWord2[uWord].uIndex;

			while ( uY != (UINT4) INFINITY4U )
			{
				uY2	= uY + WORD_SIZE - 1;
				uY2p	= uDataLen - 1 - uY2;

				SeedFindRC(&stTmpSeedList, &stRealSeedList, uX, uY2p, pbQry, pbRC, uQryLen, uDataLen);

				uY	= pstWordEle2[uY].uPrevIndex;
			}
		}
	}

	CleanTmpSeedListRC2(&stTmpSeedList, &stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);		// Empty seed in TmpSeedList to RealSeedList
	SeedNum = stRealSeedList.nSeedNum;
	CleanRealSeedListRC2(&stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);						// print RealSeedList

	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int NHitMethodRCAllow2(PWORD_ELE pstWordEle1, PTMP_WORD pstTmpWord1, PWORD_ELE pstWordEle2, PTMP_WORD pstTmpWord2, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	UINT4		j	= 0;

	UINT4		uX		= 0;
	UINT4		uY		= 0;
	UINT4		uY2		= 0;
	UINT4		uY2p	= 0;
	UINT4		uWord1	= 0;
	UINT4		uWord2	= 0;

	TMP_SEED_LIST	stTmpSeedList	= {0};
	REAL_SEED_LIST	stRealSeedList	= {0};
	
	UINT4	uAllowSeqNum	= 0;
	UINT4	SeedNum	= 0;


	InitTmpSeedList(&stTmpSeedList, (long long) uQryLen + uDataLen);
	InitRealSeedList(&stRealSeedList, REAL_SEED_NUM);


	uAllowSeqNum	= GetAllowSeqNum(ALLOW_SIZE);


	for ( i = (long long) uQryLen - 1 ; i >=0 ; --i )
	{
		TimeProgress("Seeding (RC)", uQryLen - i, uQryLen, 500000);

		uX	= pstWordEle1[i].uIndex;

		if ( uX != (UINT4) INFINITY4U )
		{
			uWord1	= GetRCWord(pbQry + uX);

			for ( j = 0 ; j < uAllowSeqNum ; ++j )
			{
				uWord2	= GetAllowSeq(uWord1, j);		// If word1 and word2 are same or similar

				uY	= pstTmpWord2[uWord2].uIndex;

				while ( uY != (UINT4) INFINITY4U )
				{
					uY2	= uY + WORD_SIZE - 1;
					uY2p	= uDataLen - 1- uY2;

					SeedFindRC(&stTmpSeedList, &stRealSeedList, uX, uY2p, pbQry, pbRC, uQryLen, uDataLen);

					uY	= pstWordEle2[uY].uPrevIndex;
				}
			}
		}
	}

	CleanTmpSeedListRC2(&stTmpSeedList, &stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);		// Empty seed in TmpSeedList to RealSeedList
	SeedNum = stRealSeedList.nSeedNum;
	CleanRealSeedListRC2(&stRealSeedList, pbQry, pbRC, uQryLen, uDataLen);						// print RealSeedList
	
	FreeTmpSeedList(&stTmpSeedList);
	FreeRealSeedList(&stRealSeedList);

	return SeedNum;
}

int SeedFindQry(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	SEED		stSEED		= {0};
	PTMP_SEED	pstTmpSeed	= NULL;

	long long	llDiag		= 0;

	int		nSpace			= 0;
	int		nIsOnlyBase		= 0;
	int		nIsSeedCombined	= 0;
	int		uLen		= 0;


	llDiag		= (long long) (uDataLen - 1 + uX - uY);

	pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[llDiag]);

	if ( pstTmpSeed->uLen )
	{
		nSpace	= pstTmpSeed->uX - ( uX + WORD_SIZE );

		if ( nSpace <= 0 )	// (1) -WORD < SPACE <= 0
		{
			WordCombine(pstTmpSeed, nSpace);
		}
		
		else if ( nSpace <= (int)SPACE_SIZE )	// (2) 0 < SPACE <= SP
		{
			nIsOnlyBase	= IsOnlyBase(pbQry + uX + WORD_SIZE, pbData + uY + WORD_SIZE, nSpace);
			
			if ( nIsOnlyBase == 1 )		// Only A,C,G,T in SPACE
			{
				WordCombine(pstTmpSeed, nSpace);
			}

			else
			{
				if ( pstTmpSeed->uLen >= MIN_SEED_LEN )
				{
					PushRealSeed(pstRealSeedList, pstTmpSeed);
				}
				UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
			}
		}
		
		else		// (3) SPACE > SP
		{
			if ( pstTmpSeed->uLen >= MIN_SEED_LEN )
			{
				PushRealSeed(pstRealSeedList, pstTmpSeed);
			}
			UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
		}
	}

	else	// no element
	{
		UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
	}


	return 1;
}

int SeedFindRC(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, UINT4 uX, UINT4 uY, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	SEED		stSEED		= {0};
	PTMP_SEED	pstTmpSeed	= NULL;

	long long	llDiag		= 0;

	int		nSpace			= 0;
	int		nIsOnlyBase		= 0;
	int		nIsSeedCombined	= 0;
	int		uLen		= 0;


	llDiag		= (long long) (uDataLen - 1 + uX - uY);

	pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[llDiag]);

	if ( pstTmpSeed->uLen )
	{
		nSpace	= pstTmpSeed->uX - ( uX + WORD_SIZE );

		if ( nSpace <= 0 )	// (1) -WORD < SPACE <= 0
		{
			WordCombine(pstTmpSeed, nSpace);
		}
		
		else if ( nSpace <= (int)SPACE_SIZE )	// (2) 0 < SPACE <= SP
		{
			nIsOnlyBase	= IsOnlyBase(pbQry + uX + WORD_SIZE, pbRC + uY + WORD_SIZE, nSpace);
			
			if ( nIsOnlyBase == 1 )		// Only A,C,G,T in SPACE
			{
				WordCombine(pstTmpSeed, nSpace);
			}

			else
			{
				if ( pstTmpSeed->uLen >= MIN_SEED_LEN )
				{
					PushRealSeed(pstRealSeedList, pstTmpSeed);
				}
				UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
			}
		}
		
		else		// (3) SPACE > SP
		{
			if ( pstTmpSeed->uLen >= MIN_SEED_LEN )
			{
				PushRealSeed(pstRealSeedList, pstTmpSeed);
			}
			UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
		}
	}

	else	// no element
	{
		UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE);
	}

	return 1;
}

int CleanTmpSeedListQry(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	
	PTMP_SEED	pstTmpSeed	= NULL;


	for ( i = 0 ; i < pstTmpSeedList->llDiagNum ; ++i )
	{
		TimeProgress("Seeding (Qry, 2/2)", i, pstTmpSeedList->llDiagNum, 5000000);

		pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[i]);

		if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Process remaining seed (gapped extension)
		{
			UngappedExt(pstTmpSeed, pbQry, pbData, uQryLen, uQryLen);

			GappedExtMgrQry(pstRealSeedList, pstTmpSeed, pbQry, pbData, uQryLen, uDataLen);
		}
	}

	return 0;
}

int CleanTmpSeedListRC(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	
	PTMP_SEED	pstTmpSeed	= NULL;


	for ( i = 0 ; i < pstTmpSeedList->llDiagNum ; ++i )
	{
		TimeProgress("Seeding (RC, 2/2)", i, pstTmpSeedList->llDiagNum, 5000000);

		pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[i]);

		if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Process remaining seed (gapped extension)
		{
			UngappedExt(pstTmpSeed, pbQry, pbRC, uQryLen, uDataLen);

			GappedExtMgrRC(pstRealSeedList, pstTmpSeed, pbQry, pbRC, uQryLen, uDataLen);
		}
	}

	return 0;
}

int CleanTmpSeedListQry2(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	
	PTMP_SEED	pstTmpSeed	= NULL;


	for ( i = 0 ; i < pstTmpSeedList->llDiagNum ; ++i )
	{
		TimeProgress("Seeding (Qry, 2/2)", i, pstTmpSeedList->llDiagNum, 5000000);

		pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[i]);

		if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Process remaining seed (gapped extension)
		{
			PushRealSeed(pstRealSeedList, pstTmpSeed);
		}
	}

	return 0;
}

int CleanTmpSeedListRC2(PTMP_SEED_LIST pstTmpSeedList, PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	long long	i	= 0;
	
	PTMP_SEED	pstTmpSeed	= NULL;


	for ( i = 0 ; i < pstTmpSeedList->llDiagNum ; ++i )
	{
		TimeProgress("Seeding (RC, 2/2)", i, pstTmpSeedList->llDiagNum, 5000000);

		pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[i]);

		if ( pstTmpSeed->uLen >= MIN_SEED_LEN )		// Process remaining seed (gapped extension)
		{
			PushRealSeed(pstRealSeedList, pstTmpSeed);
		}
	}

	return 0;
}

int WordCombine(PTMP_SEED pstTmpSeed, int nSpace)
{
	UINT4	uExtendLen	= 0;

	uExtendLen	= (UINT4) ( nSpace + (int)WORD_SIZE );

	pstTmpSeed->uX		-= uExtendLen;
	pstTmpSeed->uY		-= uExtendLen;
	pstTmpSeed->uLen	+= uExtendLen;

	return 0;
}

int SeedCombine(PTMP_SEED_LIST pstTmpSeedList, UINT4 uX, UINT4 uY, long long llDiag, char* pbSeq1, char* pbSeq2)
{
	long long	i	= 0;

	PTMP_SEED	pstTmpSeed	= NULL;
	
	UINT4	uSpace		= 0;
	UINT4	uExtNum		= 0;
	int		nScoreF		= 0;	// forward
	int		nScoreB		= 0;	// backward

	int		nIsSeedCombined	= 0;


	pstTmpSeed	= & (pstTmpSeedList->pTmpSeed[llDiag]);

	uSpace		= pstTmpSeed->uX - ( uX + WORD_SIZE );

	uExtNum		= ExtToThrForward(pbSeq1 + uX + WORD_SIZE, pbSeq2 + uY + WORD_SIZE, uSpace, &nScoreF);

	if ( uExtNum == uSpace )		// Combine seed (forward)
	{
		UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE + uSpace + pstTmpSeed->uLen);

		nIsSeedCombined	= 1;
	}
	else
	{
		uExtNum		+= ExtToThrBackward(pbSeq1 + pstTmpSeed->uX - (uSpace - uExtNum), pbSeq2 + pstTmpSeed->uY - (uSpace - uExtNum), uSpace - uExtNum, &nScoreB);

		if ( uExtNum == uSpace )	// Combine seed (backward)
		{
			UpdateTmpSeed(pstTmpSeedList, llDiag, uX, uY, WORD_SIZE + uSpace + pstTmpSeed->uLen);

			nIsSeedCombined	= 1;
		}
	}


	return nIsSeedCombined;
}

UINT4 ExtToThrForward(char* pbSeq1, char* pbSeq2, UINT4 uMaxLen, int* pnScore)
{
	UINT4	i	= 0;

	int		nScore	= 0;


	for ( i = 0 ; i < uMaxLen ; ++i )
	{
		if ( auBase2NumMap[ pbSeq1[i] - ASCII_A ] > 3 )			// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else if ( auBase2NumMap[ pbSeq2[i] - ASCII_A ] > 3 )	// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else
		{
			if ( pbSeq1[i] == pbSeq2[i] )
			{
				nScore	+= (int)SCORE_MAT;
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

	*pnScore	= nScore;

	return i;
}

UINT4 ExtToThrBackward(char* pbSeq1, char* pbSeq2, UINT4 uMaxLen, int* pnScore)
{
	UINT4	i	= 0;

	int		nScore	= 0;
	UINT4	uIndex	= 0;


	for ( i = 0 ; i < uMaxLen ; ++i )
	{
		uIndex	= uMaxLen-i-1;

		if ( auBase2NumMap[ pbSeq1[uIndex] - ASCII_A ] > 3 )		// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else if ( auBase2NumMap[ pbSeq2[uIndex] - ASCII_A ] > 3 )	// Exist anything except A, C, G, T (no base)
		{
			break;
		}
		else
		{
			if ( pbSeq1[uIndex] == pbSeq2[uIndex] )
			{
				nScore	+= (int)SCORE_MAT;
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

	*pnScore	= nScore;

	return i;
}

int IsOnlyBase(char* pbSeq1, char* pbSeq2, UINT4 uSeqLen)
{
	UINT4	i	= 0;

	int		nIsOnlyBase	= 1;


	for ( i = 0 ; i < uSeqLen ; ++i )
	{
		if ( auBase2NumMap[ pbSeq1[i] - ASCII_A ] > 3 )			// Exist anything except A, C, G, T (no base)
		{
			nIsOnlyBase	= 0;

			break;
		}
		else if ( auBase2NumMap[ pbSeq2[i] - ASCII_A ] > 3 )	// Exist anything except A, C, G, T (no base)
		{
			nIsOnlyBase	= 0;

			break;
		}
	}

	return nIsOnlyBase;
}

UINT4 GetSeedMatNum(char* pbSeq1, char* pbSeq2, UINT4 uSeqLen)
{
	UINT4	i	= 0;

	UINT4	uMatNum	= 0;


	for ( i = 0 ; i < uSeqLen ; ++i )
	{
		if ( pbSeq1[i] == pbSeq2[i] )
		{
			uMatNum	+= 1;
		}
	}

	return uMatNum;
}

UINT4 GetAllowSeqNum(UINT4 uAllowSize)
{
	UINT4	i		= 0;
	UINT4	j		= 0;
	UINT4	uTmp	= 1;

	UINT4	uAllowSeqNum	= 1;			// 0-allowable

	for ( i = 1 ; i <= uAllowSize ; ++i )	// 1~m-allowalbe
	{
		uTmp	= 1;

		for ( j = 0 ; j < i ; ++j )			// wCm
		{
			uTmp	*= ( ( WORD_SIZE - j ) / ( i - j ) );
		}

		uAllowSeqNum	+= ( uTmp * (UINT4)pow((float)3, (float)i) );
	}

	return uAllowSeqNum;
}

UINT4 GetAllowSeq(UINT4 uSeq, UINT4 uOrder)
{
	UINT4	uAllowSeq	= 0;

	UINT4	uShift	= 0;
	UINT4	uPlus	= 0;

	if ( uOrder == 0)		// 0-allowalbe
	{
		uAllowSeq	= uSeq;
	}
	else if ( uOrder < GetAllowSeqNum(1) )		// 1-allowalbe
	{
		uShift	= ( ( uOrder - 1 ) / 3 ) * 2;
		uPlus	= ( uOrder - 1 ) % 3 + 1;

		uAllowSeq	=  ( ( ( ( uSeq >> uShift ) + uPlus ) & 0x00000003 ) << uShift ) | ( uSeq & ~( 0x00000003 << uShift ) );
	}
	else		// no supported
	{
		printf("ERROR: Too big m (plz report this error to WC)\n");
		uAllowSeq	= 0;
	}

	return uAllowSeq;
}

int GappedExtMgrQry(PREAL_SEED_LIST pstRealSeedList, PTMP_SEED pstTmpSeed, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	pstTmpSeed->uMatNum	= GetSeedMatNum(pbQry + pstTmpSeed->uX, pbData + pstTmpSeed->uY, pstTmpSeed->uLen);

	PushRealSeed(pstRealSeedList, pstTmpSeed);

	if ( pstRealSeedList->nSeedNum == REAL_SEED_NUM )
	{
		GappedExtQry(pstRealSeedList, pbQry, pbData, uQryLen, uDataLen);
		
		ResetRealSeedList(pstRealSeedList);
	}

	return 0;
}

int GappedExtMgrRC(PREAL_SEED_LIST pstRealSeedList, PTMP_SEED pstTmpSeed, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	pstTmpSeed->uMatNum	= GetSeedMatNum(pbQry + pstTmpSeed->uX, pbRC + pstTmpSeed->uY, pstTmpSeed->uLen);

	PushRealSeed(pstRealSeedList, pstTmpSeed);

	if ( pstRealSeedList->nSeedNum == REAL_SEED_NUM )
	{
		GappedExtRC(pstRealSeedList, pbQry, pbRC, uQryLen, uDataLen);
		
		ResetRealSeedList(pstRealSeedList);
	}

	return 0;
}

int CleanRealSeedListQry(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	GappedExtQry(pstRealSeedList, pbQry, pbData, uQryLen, uDataLen);
	
	ResetRealSeedList(pstRealSeedList);

	return 0;
}

int CleanRealSeedListRC(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	GappedExtRC(pstRealSeedList, pbQry, pbRC, uQryLen, uDataLen);
	
	ResetRealSeedList(pstRealSeedList);

	return 0;
}

int CleanRealSeedListQry2(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbData, UINT4 uQryLen, UINT4 uDataLen)
{
	UINT4	i	= 0;

	SEED		stSEED		= {0};
	TMP_SEED	stTmpSeed	= {0};

	for ( i = 0 ; i < pstRealSeedList->nSeedNum ; ++i )
	{
		stTmpSeed		= GetRealSeed(pstRealSeedList, i);
		stSEED.uX1		= stTmpSeed.uX;
		stSEED.uX2		= stTmpSeed.uX + stTmpSeed.uLen - 1;
		stSEED.uY1		= stTmpSeed.uY;
		stSEED.uY2		= stTmpSeed.uY + stTmpSeed.uLen - 1;
		//printf("%d, %d, %d, %d\n",stSEED.uX1,stSEED.uX2,stSEED.uY1,stSEED.uY2);
		WriteSeedFileQry(&stSEED,uQryLen,uDataLen);
	}

	
	ResetRealSeedList(pstRealSeedList);

	return 0;
}

int CleanRealSeedListRC2(PREAL_SEED_LIST pstRealSeedList, char* pbQry, char* pbRC, UINT4 uQryLen, UINT4 uDataLen)
{
	UINT4	i	= 0;

	SEED		stSEED		= {0};
	TMP_SEED	stTmpSeed	= {0};

	for ( i = 0 ; i < pstRealSeedList->nSeedNum ; ++i )
	{
		stTmpSeed		= GetRealSeed(pstRealSeedList, i);
		stSEED.uX1		= stTmpSeed.uX;
		stSEED.uX2		= stTmpSeed.uX + stTmpSeed.uLen - 1;
		stSEED.uY1		= stTmpSeed.uY;
		stSEED.uY2		= stTmpSeed.uY + stTmpSeed.uLen - 1;
		//printf("%d, %d, %d, %d\n",stSEED.uX1,stSEED.uX2,stSEED.uY1,stSEED.uY2);
		WriteSeedFileRC(&stSEED,uQryLen,uDataLen);
	}

	
	ResetRealSeedList(pstRealSeedList);

	return 0;
}

int WriteSeedFileQry(PSEED pstSEED, UINT4 uQryLen, UINT4 uDataLen)
{
#pragma omp critical (file_lock)
	{
		fwrite(pstSEED, sizeof(SEED), 1, g_pfRes);
	}

	return 0;
}

int WriteSeedFileRC(PSEED pstSEED, UINT4 uQryLen, UINT4 uDataLen)
{
	pstSEED->uY1	= uDataLen - pstSEED->uY1 - 1;
	pstSEED->uY2	= uDataLen - pstSEED->uY2 - 1;

#pragma omp critical (file_lock)
	{
		fwrite(pstSEED, sizeof(SEED), 1, g_pfRes);
	}

	return 0;
}
