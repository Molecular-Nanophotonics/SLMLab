/*   
   Copyright 2009, 2010, 2011 Boulder Nonlinear Systems 
   ALinnenberger@bnonlinear.com

   This file is part of GenerateHologramCUDA.

    GenerateHologramCUDA is free software: you can redistribute it and/or 
    modify it under the terms of the GNU Lesser General Public License as published 
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GenerateHologramCUDA is distributed in the hope that it will be 
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with GenerateHologramCUDA.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "stdafx.h"
#include "math.h"
#include "PCIeBoard.h"

#include <fstream>
#include <direct.h>

//global variable
CPCIeBoard* theBoard;

//local function
bool ReadLUTFile(unsigned char* LUT, CString LUTFileName);

typedef struct{
    float x, y;
} Complex;



////////////////////////////////////////////////////////////////////////////////
// FUNCTIONS TO TALK TO THE PCIe HARDWARE
////////////////////////////////////////////////////////////////////////////////

extern "C" int InitalizeSLM(bool bRAMWriteEnable, unsigned short TrueFrames)
{
	char buffer[_MAX_PATH];
	char buffer2[_MAX_PATH];
	bool Init = true;
	bool TestEnable = false;
	unsigned char LUT[256];
	theBoard = new CPCIeBoard(&buffer[0],&buffer2[0], Init, TestEnable, bRAMWriteEnable);
	theBoard->SetTrueFrames(TrueFrames);
	
	//load a linear LUT to hardware
	for (int ii = 0;ii<256;ii++)
		LUT[ii] = ii;
	theBoard->WriteLUT(LUT);

	return GetLastError();
}

extern "C" void LoadImg(unsigned char* Img)
{
	theBoard->WriteFrameBuffer(Img);
}

extern "C" void Wait(int Delay_ms)
{
	__int64 frequency = 0;
	QueryPerformanceFrequency((LARGE_INTEGER *)&frequency);
	__int64 startTime, stopTime;

	QueryPerformanceCounter((LARGE_INTEGER *)&startTime);
	QueryPerformanceCounter((LARGE_INTEGER *)&stopTime);

	while((((double)stopTime - startTime) / frequency) * 1000 < Delay_ms)
		QueryPerformanceCounter((LARGE_INTEGER *)&stopTime);
}

extern "C" void SetPower(bool bPower)
{
	theBoard->SetPower(bPower);
}

extern "C" void ShutDownSLM()
{
	theBoard->SetPower(0);
	delete theBoard;
}

bool ReadLUTFile(unsigned char* LUT, CString LUTFileName)
{
	int i, seqnum, scanCount, tmpLUT;
	char inbuf[_MAX_PATH];
	bool errorFlag = false;

	std::ifstream LUTFile(LUTFileName);
	if (!LUTFile.is_open())
	{	
		AfxMessageBox("LUT File is currently open - could not open file");
		return false;
	}

	//read in each line of the file
	i = 0;
	while (LUTFile.getline(inbuf, _MAX_PATH, '\n'))
	{
		// parse out the Image Entries
		scanCount=sscanf(inbuf, "%d %d", &seqnum, &tmpLUT);

		//if scanCount is 0, then no fields were assigned. This is an invalid row
		//Or if scanCount does not equal 15, then there were not 15 items in that
		//row. This would indicate an invalid row. In either case we "continue" to break
		//out of the while loop
		if ((scanCount != 2)||(seqnum != i)||(tmpLUT < 0) || (tmpLUT > 255))
		{
			errorFlag = true;		
			AfxMessageBox("Error Reading LUT File");
			continue;
		}

		LUT[seqnum] = (unsigned char) tmpLUT;
		i++;
	}
	LUTFile.close();

	if (errorFlag == true)                    
	{
		for (i=0; i<256; i++)
			LUT[i]=i;
		return false;
	}

	return TRUE;
}
