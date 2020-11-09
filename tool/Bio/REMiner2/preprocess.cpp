#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <memory.h>
#include "GVD.H"
#include "preprocess.h"
#include "timelog.h"

int CaseMatch(char* pbSeq, UINT4 uDataLen)
{
	UINT4	i	= 0;

	for ( i = 0 ; i < uDataLen ; ++i )
	{
		TimeProgress("Case Matching", i, uDataLen, 10000000);

		pbSeq[i] = (char) toupper (pbSeq[i]);
	}

	return 0;
}

int Filtering(char* pbOutput, const char* pbInput, UINT4 uDataLen)
{
	UINT4	i	= 0;

	int		anBaseCnt[4]	= {0};
	int		nMaxNoBaseIndex	= 0;
	float	fScore			= (float)0;


	memcpy(pbOutput, pbInput, uDataLen * sizeof(char));

	while ( i < uDataLen - WD_SIZE + 1 )
	{
		TimeProgress("Filtering", i, uDataLen, 10000000);

		memset(anBaseCnt, 0, sizeof(anBaseCnt));

		nMaxNoBaseIndex	= GetBaseCnt(pbInput+i, WD_SIZE, anBaseCnt);

		if ( nMaxNoBaseIndex )		// no base
		{
			i	+= ( nMaxNoBaseIndex + 1 );		// Jump non-A,C,G,T and Scan the bases
		}
		else
		{
			fScore	= GetScore(anBaseCnt, WD_SIZE);

			if ( (fScore + 0.001) >= T_THR )
			{
				Mask(pbOutput, pbInput, i, i + WD_SIZE - 1);
			}

			i	+= 1;
		}
	}

	return 0;
}

int RC_Conversion(char* pbOutput, const char* pbInput, UINT4 uDataLen)
{
	UINT4	i	= 0;

	for ( i = 0 ; i < uDataLen ; ++i )
	{
		TimeProgress("RC Conversion", i, uDataLen, 10000000);

		pbOutput[i] = abComplementMap [ pbInput[uDataLen - 1 - i] - ASCII_A ];
	}

	return 0;
}

int Mask(char* pbOutput, const char* pbInput, UINT4 uStart, UINT4 uEnd)
{
	UINT4	i	= 0;

	for ( i = uStart ; i <= uEnd ; ++i)
	{
		pbOutput[i] = abMaskMap[ pbInput[i] - ASCII_A ];
	}

	return 0;
}

int GetBaseCnt(const char* pbSeq, int nSeqLen, int* pnBaseCnt)
{
	int	i	= 0;

	int	nMaxNoBaseIndex	= 0;
	int	nCurBaseNum		= 0;

	for ( i = nSeqLen - 1 ; i >= 0 ; --i )
	{
		nCurBaseNum	= auBase2NumMap[ pbSeq[i] - ASCII_A ];

		if ( nCurBaseNum > 3 )		// no base
		{
			nMaxNoBaseIndex	= i;

			break;
		}
		else
		{
			pnBaseCnt[nCurBaseNum]	= pnBaseCnt[nCurBaseNum] + 1;
		}
	}

	return nMaxNoBaseIndex;
}

float GetScore(int* pnBaseCnt, int nLen)
{
	float	fScore		= (float)0;
	int		nMaxBaseCnt	= 0;

	nMaxBaseCnt	= pnBaseCnt[0];			// A

	if ( pnBaseCnt[1] > nMaxBaseCnt )
	{
		nMaxBaseCnt	= pnBaseCnt[1];		// C
	}

	if ( pnBaseCnt[2] > nMaxBaseCnt )
	{
		nMaxBaseCnt	= pnBaseCnt[2];		// G
	}

	if ( pnBaseCnt[3] > nMaxBaseCnt )
	{
		nMaxBaseCnt	= pnBaseCnt[3];		// T
	}

	fScore	= (float) nMaxBaseCnt / nLen;

	return fScore;
}
