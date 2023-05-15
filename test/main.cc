#include "mkl.h"
#include <vector>

int main() {

    std::size_t n = 20;
    std::vector<double> O = std::vector<double>(n*n); 
    std::vector<double> d = std::vector<double>(n); 

    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, &O[0], n, &d[0]);

    return 0;
}
