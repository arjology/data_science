#ifndef DEF_PBAR
#define DEF_PBAR
#define PBWIDTH 60

#include <iostream>

void pbar(double pct);

#endif


void pbar(double pct)
{
    std::cout << "[";
    int pos = PBWIDTH * pct;
    std::string output;
    for (int i = 0; i < PBWIDTH; ++i)
    {
        output = i < pos ? "=" : i == pos ? ">" : " ";
        std::cout << output;
    }
    std::cout << "] " << int(pct * 100.0) << " %\r";
    std::cout.flush();

}