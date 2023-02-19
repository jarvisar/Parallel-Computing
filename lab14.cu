#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/generate.h> 
#include <thrust/sort.h> 
#include <thrust/copy.h> 
#include <thrust/unique.h>
#include <thrust/extrema.h> 
#include <cstdlib> 
#include <thrust/count.h>
#include <thrust/binary_search.h>
#include <thrust/inner_product.h>
#include <iomanip>

struct rand_functor {
const int a; 
rand_functor(int _a) : a(_a) {} 
__host__ __device__ 
int  operator()() const { 
return rand() % a; 
}}; 

template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

int main(void) { 

for (int n =2; n< (2<<20); n *=2){ 

thrust::host_vector<int> hv(n); 

thrust::generate(hv.begin(), hv.end(), rand_functor(n)); 

thrust::device_vector<int> dv = hv; 
thrust::sort(dv.begin(), dv.end()); 
thrust::counting_iterator<int> search_begin(0);

thrust::device_vector<int> B(n);
thrust::device_vector<int> C(n);

thrust::reduce_by_key(dv.begin(), dv.end(), thrust::constant_iterator<int>(1), B.begin(), C.begin());

thrust::device_vector<int>::iterator iter = 
thrust::max_element(C.begin(), C.end());

unsigned int position = iter - dv.begin(); 
float max_val = *iter;

printf("Max bin capacity for N = %d bins is: %.0f \n",n, max_val);
} 
return 0; 
}

