#include "sample/sampler.hpp"
#include <fstream>
#include <iostream>
int main(int argc, char** argv){
 if(argc != 2) { std::cerr << "usage: replay_sample LOGITS\n"; return 2; }
 std::ifstream in(argv[1]);
 if(!in) return 2;
 std::vector<int32_t> context;
 uint64_t s=6500;
 for(int i=0;i<155;++i){s^=s<<13;s^=s>>7;s^=s<<17;context.push_back(s%512);}
 for(int trial=0;trial<2;++trial){
  std::vector<float> v(512);for(auto& f:v)in>>f;
  dgpp::sample::Params p;p.temperature=100;p.presence_penalty=.2f;
  dgpp::sample::Rng rng{4567,0};
  auto result=dgpp::sample::sample_full_logits(v.data(),512,{{0,256},{256,256}},p,rng,context);
  std::cout<<trial<<" token="<<result.token<<" counter="<<rng.counter<<"\n";
 }
}
