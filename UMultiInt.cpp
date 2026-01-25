#pragma GCC target("fma")
#include<vector>
#include<stdexcept>
#include<cmath>
#include<memory>
#include<cstring>
#include<iostream>
#include<string>
#include<climits>
#include<immintrin.h>
#include<complex>
#include<random>
#include<algorithm>
#define lf double
#define ull unsigned long long
#define ll long long
#define u32 unsigned
#define int128 __int128_t
#define __AVX2__ 1
#ifdef SIZE
#define LENGTH SIZE
#else
#define LENGTH 4000004
#endif

namespace ERROR{
	class Exception:public std::exception{
		std::string message;
	public:
		Exception(const std::string&c):message(c){}
		const char* what()const noexcept override{return message.c_str();}
	};
	class Div_by_zero:public Exception{
	public:
		Div_by_zero():Exception("Error:Division by zero!"){}
		Div_by_zero(const std::string&s):Exception("Error:"+s+" Division by zero!"){}
	};
	class MLE:public Exception{
	public:
		MLE():Exception("Error:Memory Limit Exceeded!"){}
		MLE(const std::string&s):Exception("Error:"+s+" is Memory Limit Exceeded!"){}
	};
	class Negative:public Exception{
	public:
		Negative(const std::string&s):Exception("Error:"+s+" is negative!"){}
	};
	class Number:public Exception{
	public:
		Number():Exception("Error:Not a valid numeric string!"){}
	};
	class Out_of_range:public Exception{
	public:
		Out_of_range():Exception("Error:Out of range!"){}
	};
}

namespace Transform{
#ifndef __AVX2__
	template<typename T>
	struct fComplex{
		T rez,imz;
		fComplex(){rez=0.0,imz=0.0;}
		fComplex(T x,T y){rez=x,imz=y;}
		template<class C>fComplex(std::complex<C>x):rez(x.real()),imz(x.imag()){}
		fComplex operator+(const fComplex&b)const{return {rez+b.rez,imz+b.imz};}
		fComplex operator-(const fComplex&b)const{return {rez-b.rez,imz-b.imz};}
		fComplex operator*(const fComplex&b)const{return {rez*b.rez-imz*b.imz,rez*b.imz+imz*b.rez};}
		fComplex& operator*=(const fComplex&b){*this=*this*b;return *this;}
		fComplex& operator*(const T&b){return {rez*b,imz*b};}
		fComplex& operator*=(const T&b){rez*=b,imz*=b;return *this;}
		fComplex operator+(const T&x)const{return rez+=x;}
		fComplex conj()const{return {rez,-imz};}
		T real()const{return rez;}
		T imag()const{return imz;}
	};
	using Complex=fComplex<lf>;
	struct FFT{
		const lf pi=3.141592653589793;
		const lf pi2=6.283185307179586;
		std::vector<Complex>fft_a;
		void init(int len){fft_a.resize(len);}
		std::vector<int>rev;std::vector<Complex>omega;
		void fft(int flag,int len){
			for(int i=0;i<len;i++){
				if(i<rev[i]) std::swap(fft_a[i],fft_a[rev[i]]);
			}
			for(int i=2;i<=len;i<<=1){
				int t=len/i;
				for(int j=0;j<len;j+=i){
					for(int k=0;k<i/2;k++){
						int idx=k*t;
						if(flag<0) idx=len-idx;
						if(idx>=len) idx-=len;
						Complex w=omega[idx];
						Complex x=fft_a[j+k];
						Complex y=w*fft_a[j+k+i/2];
						fft_a[j+k]=x+y;
						fft_a[j+k+i/2]=x-y;
					}
				}
			}
			lf inv=1/len;
			if(flag==-1) for(int i=0;i<len;i++) fft_a[i]*=inv;
		}
		void Init(int k){
			rev.resize(1<<k,0);
			int l=1<<k;
			for(int i=0;i<l;i++) rev[i]=(rev[i>>1]>>1)|((i&1)<<(k-1));
			omega.clear(),omega.resize(l,{0.0,0.0});
			omega[0]={1.0,0.0};
			for(int i=1;i<l;i<<=1) omega[i]={cos(pi2*i/l),sin(pi2*i/l)};
			for(int i=0;i<l;i++) if(i&(i-1)) omega[i]=omega[i&(-i)]*omega[i&(i-1)];
		}
	};
#else
	struct Complex{
		__m128d val;
		Complex()=default;
		Complex(const __m128d&x):val(x){}
		Complex(lf x,lf y):val(_mm_set_pd(y,x)){}
		template<class C>Complex(std::complex<C>x):val(_mm_set_pd(x.imag(),x.real())){}
		Complex operator+(const Complex&x)const{return _mm_add_pd(val,x.val);}
		Complex operator-(const Complex&x)const{return _mm_sub_pd(val,x.val);}
		Complex operator*(const Complex&x)const{return _mm_fmaddsub_pd(_mm_unpacklo_pd(val,val),x.val,_mm_unpackhi_pd(val,val)*_mm_permute_pd(x.val,1));}
		Complex& operator*=(const Complex&b){val=_mm_fmaddsub_pd(_mm_unpacklo_pd(val,val),b.val,_mm_unpackhi_pd(val,val)*_mm_permute_pd(b.val,1));return *this;}
		Complex operator*(lf x)const{return _mm_mul_pd(val,_mm_set1_pd(x));}
		Complex& operator*=(lf x){val=_mm_mul_pd(val,_mm_set1_pd(x));return *this;}
		Complex operator+(lf x)const{return _mm_add_pd(val,_mm_set_pd(0.0,x));}
		Complex conj()const{return Complex(_mm_xor_pd(val,_mm_set_pd(-0.0,0.0)));}
		Complex operator-()const{return _mm_mul_pd(val,_mm_set1_pd(-1.0));}
		lf real()const{return _mm_cvtsd_f64(val);}
		lf imag()const{return _mm_cvtsd_f64(_mm_unpackhi_pd(val,val));}
	};
	struct FFT{
		const lf pi=3.141592653589793;
		const lf pi2=6.283185307179586;
		std::vector<Complex>omega;
		Complex calc(const Complex&a,const Complex&b){return _mm_fmadd_pd(_mm_unpacklo_pd(a.val,a.val),b.val,_mm_unpackhi_pd(a.val,a.val)*_mm_permute_pd(b.val,1));}
		void init(int Len){
			if(Len<=(int)omega.size()<<1) return;
			int k=std::__lg(Len-1);
			omega.resize(1<<k),omega[0]={1.0,0.0};Len=1<<k;
			for(int i=1;i<Len;i<<=1) omega[i]=std::polar(1.0,pi/(i<<1));
			for(int i=1;i<Len;i++) if(i&(i-1)) omega[i]=omega[i&(-i)]*omega[i&(i-1)];
		}
		void dif(std::vector<Complex>&a){
			int len=a.size();
			for(int Len=len>>1,sp=len;Len;sp=Len,Len>>=1){
				for(int i=0;i<Len;i++){auto temp=a[i];a[i]=temp+a[i+Len],a[i+Len]=temp-a[i+Len];}
				for(int blk=sp,o=1;blk<len;blk+=sp,o++){
					for(int i=blk;i<blk+Len;i++){auto t1=a[i],t2=a[Len+i]*omega[o];a[i]=t1+t2,a[Len+i]=t1-t2;}
				}
			}
		}
		void dit(std::vector<Complex>&a){
			int len=a.size();
			for(int Len=1,sp=2;Len!=len;Len=sp,sp<<=1){
				for(int i=0;i<Len;i++){auto temp=a[i];a[i]=temp+a[i+Len],a[i+Len]=temp-a[i+Len];}
				for(int blk=sp,o=1;blk<len;blk+=sp,o++){
					for(int i=blk;i<blk+Len;i++){auto t1=a[i],t2=a[Len+i];a[i]=t1+t2,a[Len+i]=(t1-t2)*omega[o].conj();}
				}
			}
		}
		void mul(std::vector<Complex>&F,std::vector<Complex>&G){
			int len=F.size();
			lf inv=1.0/len,_2=inv*0.25;
			F[0]=calc(F[0],G[0])*inv;
			F[1]=F[1]*G[1]*inv;
			for(int st=2,ed=3;st<len;st<<=1,ed<<=1){
				for(int i=st,j=i+st-1;i<ed;i++,j--){
					Complex oi=(F[i]+F[j].conj()),hi=(F[i]-F[j].conj());
					Complex Oi=(G[i]+G[j].conj()),Hi=(G[i]-G[j].conj());
					Complex A=oi*Oi-hi*Hi*((i&1)?-omega[i>>1]:omega[i>>1]),B=Oi*hi+oi*Hi;
					F[i]=(A+B)*_2,F[j]=(A-B).conj()*_2;
				}
			}
		}
	};
#endif
}


template<u32 _RADIX=10,u32 _LEN=8,u32 _BASE=100000000,u32 _FFT_BASE=10000>
class UMultiInt{
private:
	int c_i(char c)const{return (isdigit(c)?c-'0':(isupper(c)?c-'A'+10:c-'a'+36));}
	int i_c(int i)const{return ((0<=i&&i<=9)?i+'0':((10<=i&&i<=35)?(i-10)+'A':(i-36)+'a'));}
	u32 log(ull p)const{u32 cnt=0;while(p){cnt++,p/=RADIX;}return cnt;}
	void i_to_S(std::string&s,ull t)const{
		size_t size=log(t);
		s.resize(s.size()+size);int cnt=0;
		while(t){s[size-1-cnt]=i_c(t%RADIX);t/=RADIX;cnt++;}
	}
	static const u32 RADIX=_RADIX,LEN=_LEN,BASE=_BASE,FFT_BASE=_FFT_BASE,T=255,INV=64,DEFAULT=1;
	static const ull MAX=ULLONG_MAX/(BASE+1),LOG=std::__lg(MAX)-1;
	std::unique_ptr<ull[]>num;int len,Max;
	bool is_zero()const{return len==1&&num[0]==0;}
	void Expand(int nMax){
		if(nMax<=Max) return;
		int _Max=std::max(Max<<1,nMax);
		if(_Max>LENGTH) throw ERROR::MLE("This number");
		auto _num=std::make_unique<ull[]>(_Max);
		if(num) std::copy(num.get(),num.get()+len,_num.get());
		num=std::move(_num);
		Max=_Max;
	}
	void Mul(const UMultiInt&b){
		ull* temp=new ull[len+b.len+5]();
		for(int i=0;i<len;i++){
			ull carry=0,t=num[i];
			for(int j=0;j<b.len;j++){carry+=temp[i+j]+t*b.num[j];temp[i+j]=carry%BASE,carry/=BASE;}
			temp[i+b.len]+=carry;
		}
		Expand(len+b.len+5);len=len+b.len+5;
		std::fill(num.get(),num.get()+len,0);
		for(int i=0;i<len;i++) num[i]=temp[i];
		for(int i=0;i<len;i++) num[i+1]+=num[i]/BASE,num[i]%=BASE;
		while(len>1&&num[len-1]==0) len--;
		delete[] temp;
	}
	ull Get(const UMultiInt&a,int pos)const{
		return 10ull*BASE*(pos+1>=a.len?0:a.num[pos+1])+10ull*a.num[pos]+(pos?a.num[pos-1]:0)/(BASE/10);
	}
	std::pair<UMultiInt,UMultiInt> Simple_Mod(const UMultiInt&b)const{
		if(b.is_zero()) throw ERROR::Div_by_zero();
		if(*this<b) return std::make_pair(0,*this);
		if(*this==b) return std::make_pair(1,0);
		if(b.len<=2){ull q=b.num[0]+b.num[1]*BASE;return std::make_pair(*this/q,*this%q);}
		UMultiInt Q,R(*this);Q.Expand(len-b.len+1),Q.len=len-b.len+1;
		ull b_Get=Get(b,b.len-1)+1;
		for(int i=len-b.len;i>=0;i--){
			ull q=0;
			auto Sub=[&](){
				ll t=0;
				for(int j=0;j<b.len;j++){
					t=t-q*b.num[j]+R.num[i+j];
					R.num[i+j]=(ull)(t%BASE),t/=BASE;
					if(R.num[i+j]>=BASE) R.num[i+j]+=BASE,t--;
				}
				if(t) R.num[i+b.len]+=(ull)(t);
				Q.num[i]+=q;
			};
			while((q=Get(R,i+b.len-1)/b_Get)) Sub();
			q=1;
			for(int j=b.len-1;j>=0;j--){if(R.num[j+i]!=b.num[j]&&(q=b.num[j]<R.num[i+j],true)) break;}
			if(q) Sub();
		}
		while(Q.len>1&&Q.num[Q.len-1]==0) Q.len--;
		while(R.len>1&&R.num[R.len-1]==0) R.len--;
		return std::make_pair(Q,R);
	}
	UMultiInt Left(int cnt)const{
		if(cnt<0) throw ERROR::Negative("Left Shift count");
		if(len+cnt>LENGTH) throw ERROR::MLE("Left Shift");
		if(cnt==0||is_zero()) return *this;
		UMultiInt res;
		res.Expand(len+cnt),res.len=len+cnt;
		std::copy(num.get(),num.get()+len,res.num.get()+cnt);
		while(res.len>1&&res.num[res.len-1]==0) res.len--;
		return res;
	}
	UMultiInt Right(int cnt)const{
		if(cnt<0) throw ERROR::Negative("Right Shift count");
		if(cnt>=len) return UMultiInt();
		UMultiInt res;
		res.Expand(len-cnt);res.len=len-cnt;
		std::copy(num.get()+cnt,num.get()+len,res.num.get());
		while(res.len>1&&res.num[res.len-1]==0) res.len--;
		return res;
	}
	UMultiInt Inv(int n)const{
		if(is_zero()) throw ERROR::Div_by_zero();
		if(len<=(int)INV||n<=(int)INV+len){
			UMultiInt a;a.Expand(n+1),a.len=n+1;
			std::fill(a.num.get(),a.num.get()+a.len,0),a.num[n]=1;
			return a.Simple_Mod(*this).first;
		}
		int k=(n-len+5)>>1,kk=k>len?0:len-k;
		UMultiInt t=Right(kk);
		int n1=k+t.len;
		UMultiInt t1=t.Inv(n1);
		UMultiInt res=(t1+t1).Left(n-n1-kk)-(*this*t1*t1).Right(2*(n1+kk)-n);
		return --res;
	}
	std::pair<UMultiInt,UMultiInt> Mod(const UMultiInt&b)const{
		//https://judge.yosupo.jp/submission/343251
		if(*this<b) return std::make_pair(0,*this);
		if(len<=(int)T||b.len<=(int)T) return Simple_Mod(b);
		int Len=len-b.len+5,cnt=Len>b.len?0:b.len-Len;
		UMultiInt tem=b.Right(cnt);
		if(cnt) tem++;
		int inv=Len+tem.len;
		UMultiInt Q=(*this*tem.Inv(inv)).Right(inv+cnt);
		while(Q*b>*this) Q--;
		UMultiInt R=*this-Q*b;
		while(R>=b) Q++,R-=b;
		return std::make_pair(Q,R);
	}
	ull Get_sqrt()const{
		if(len<2) throw ERROR::Out_of_range();
		int top=len-1;ull res=num[top]*BASE+num[top-1];
		return std::sqrt(res)+1;
	}
public:
	UMultiInt():len(1),Max(DEFAULT){
		if(RADIX>62) throw ERROR::Number();
		num=std::make_unique<ull[]>(DEFAULT);
	}
	~UMultiInt()=default;
	UMultiInt(const UMultiInt&b):len(b.len),Max(b.Max){
		num=std::make_unique<ull[]>(Max);
		std::copy(b.num.get(),b.num.get()+len,num.get());
	}
	UMultiInt(UMultiInt&&b)noexcept:num(std::move(b.num)),len(b.len),Max(b.Max){
		b.len=1;
		b.Max=DEFAULT;
		b.num=std::make_unique<ull[]>(DEFAULT);
	}
	UMultiInt(const ull&b):len(0),Max(DEFAULT){
		num=std::make_unique<ull[]>(DEFAULT);
		ull x=b;do{num[len]=x%BASE,len++;}while(x/=BASE);
		while(len>1&&!num[len-1]) len--;
	}
	UMultiInt& operator=(const UMultiInt&b){
		if(this!=&b){
			if(Max<b.len) num.reset(),num=std::make_unique<ull[]>(b.len),Max=b.len;
			std::copy(b.num.get(),b.num.get()+b.len,num.get());len=b.len;
		}
		return *this;
	}
	UMultiInt& operator=(UMultiInt&&b)noexcept{
		if(this!=&b){
			num=std::move(b.num),len=b.len,Max=b.Max;
			b.len=1,b.Max=DEFAULT,b.num=std::make_unique<ull[]>(DEFAULT);
		}
		return *this;
	}
	void init(const u32&size=DEFAULT){
		Expand(size),len=1;
		std::fill(num.get(),num.get()+size,0);
	}
	UMultiInt(const std::string&s){
		if(s.empty()) throw ERROR::Number();
		if(s[0]=='-') throw ERROR::Negative("This number");
		for(int i=0;i<(int)s.size();i++) if(c_i(s[i])>(int)RADIX) ERROR::Number();
		len=(s.size()+LEN-1)/LEN;
		if(len>=LENGTH) throw ERROR::MLE("This number");
		Max=std::max((int)DEFAULT,len+5),num=std::make_unique<ull[]>(Max);
		u32 base=1;
		for(int i=(int)s.size()-1,j=0;i>=0;i--,j++){
			num[j/LEN]+=c_i(s[i])*base;base*=RADIX;
			if(base==BASE) base=1;
		}
		while(len>1&&num[len-1]==0) len--;
	}
	UMultiInt(const char*s){
		u32 Len=std::strlen(s);
		if(Len==0) throw ERROR::Number();
		if(s[0]=='-') throw ERROR::Negative("This number");
		for(int i=0;i<(int)Len;i++) if(c_i(s[i])>RADIX) throw ERROR::Number();
		len=(Len+LEN-1)/LEN;
		if(len>=LENGTH) throw ERROR::MLE("This number");
		Max=std::max((int)DEFAULT,len+5),num=std::make_unique<ull[]>(Max);
		u32 base=1;
		for(int i=(int)len-1,j=0;i>=0;i--,j++){
			num[j/LEN]+=c_i(s[i])*base;base*=RADIX;
			if(base==BASE) base-=BASE;
		}
		while(len>1&&num[len-1]==0) len--;
	}
#if __cplusplus>=201703L
	UMultiInt(std::string_view s){
		if(s.empty()) throw ERROR::Number();
		if(s[0]=='-') throw ERROR::Negative("This number");
		for(int i=0;i<(int)s.size();i++) if(c_i(s[i])>RADIX) throw ERROR::Number();
		len=(s.size()+LEN-1)/LEN;
		if(len>=LENGTH) throw ERROR::MLE("This number");
		Max=std::max((int)DEFAULT,len+5),num=std::make_unique<ull[]>(Max);
		for(int i=(int)s.size()-1,j=0;i>=0;i--,j++){
			num[j/LEN]+=c_i(s[i])*j;j*=RADIX;
			if(j==BASE) j-=BASE;
		}
		while(len>1&&num[len-1]==0) len--;
	}
#endif
	friend std::istream& operator>>(std::istream&in,UMultiInt&x){
		std::string s;in>>s;
		if(in) x=UMultiInt(s);
		return in;
	}
	friend std::ostream& operator<<(std::ostream&out,const UMultiInt&x){
		if(RADIX==10){
			out<<x.num[x.len-1];
			for(int i=x.len-2;i>=0;i--){
				char buf[10];
				sprintf(buf,"%08llu",x.num[i]);
				out<<buf;
			}
			return out;
		}
		std::string s="";x.i_to_S(s,x.num[x.len-1]);
		out<<s;
		for(int i=x.len-2;~i;i--){
			ull t=x.num[i];s="";
			for(u32 j=1;j<=LEN-x.log(t);j++) s+='0';
			x.i_to_S(s,x.num[i]);
			out<<s;
		}
		return out;
	}
	int Two(){
		if(num[0]%2){return 0;}
		*this/=2;
		return Two()+1;
	}
	int Cmp(const UMultiInt&b)const{
		if(len!=b.len) return len>b.len?1:-1;
		for(int i=len-1;i>=0;i--){
			if(num[i]!=b.num[i]) return num[i]>b.num[i]?1:-1;
		}
		return 0;
	}
#if __cplusplus>=202002L
	std::strong_ordering operator<=>(const UMultiInt&b)const{return Cmp(b)<=>0;}
#endif
	bool operator<(const UMultiInt&b)const{return Cmp(b)<0;}
	bool operator>=(const UMultiInt&b)const{return Cmp(b)>=0;}
	bool operator<=(const UMultiInt&b)const{return Cmp(b)<=0;}
	bool operator>(const UMultiInt&b)const{return Cmp(b)>0;}
	bool operator==(const UMultiInt&b)const{return Cmp(b)==0;}
	bool operator!=(const UMultiInt&b)const{return !(*this==b);}
	ull operator[](const int&b)const{return num[b];}
	ull at(const int&b)const{if(b>=Max||b<0) throw ERROR::Out_of_range();return num[b];}
	UMultiInt operator+(const UMultiInt&b)const{
		UMultiInt c(*this);c+=b;
		return c;
	}
	UMultiInt& operator+=(const UMultiInt&b){
		int n=std::max(len,b.len)+1;Expand(n),len=n;
		for(int i=0;i<n;i++) num[i]+=((i<b.len)?b.num[i]:0);
		for(int i=0;i<n;i++){
			if(num[i]>=BASE){
				num[i+1]+=num[i]/BASE;num[i]%=BASE;
			}
		}
		while(len>1&&num[len-1]==0) len--;
		return *this;
	}
	UMultiInt& operator++(){return *this+=1;}
	UMultiInt operator++(int){UMultiInt res(*this);return *this+=1,res;}
	UMultiInt operator-(const UMultiInt&b)const{
		UMultiInt c(*this);
		c-=b;return c;
	}
	UMultiInt& operator-=(const UMultiInt&b){
		if(*this<b) throw ERROR::Negative("Result");
		ull t=0;
		for(int i=0;i<len;i++){
			ull sub=((i<b.len)?b.num[i]:0)+t;
			if(num[i]<sub) num[i]=BASE+num[i]-sub,t=1;
			else num[i]-=sub,t=0;
			if(t==0&&i>=b.len) break;
		}
		while(len>1&&num[len-1]==0) len--;
		return *this;
	}
	UMultiInt& operator--(){return *this-=1;}
	UMultiInt operator--(int){UMultiInt res(*this);return *this-=1,res;}
	UMultiInt operator*(const UMultiInt&b)const{
		UMultiInt c(*this);c*=b;
		return c;
	}
	UMultiInt& operator*=(const UMultiInt&b){
		//https://judge.yosupo.jp/submission/343248
		if(len<=(int)T||b.len<=(int)T){Mul(b);return *this;}
#ifdef __AVX2__
		int _len=len+b.len,Len=2<<std::__lg(_len-1);
		if(Len>LENGTH) throw ERROR::MLE("FFT Length");
		Transform::FFT H;
		std::vector<Transform::Complex>F(Len,{0.0,0.0}),G(Len,{0.0,0.0});
		for(int i=0;i<len;i++) F[i]={(lf)(num[i]%FFT_BASE),(lf)(num[i]/FFT_BASE)};
		for(int i=0;i<b.len;i++) G[i]={(lf)(b.num[i]%FFT_BASE),(lf)(b.num[i]/FFT_BASE)};
		H.init(Len);H.dif(F),H.dif(G),H.mul(F,G),H.dit(F);
		std::fill(num.get(),num.get()+len,0);
		Expand(Len);len=Len;
		ull carry=0;
		for(int i=0;i<Len;i++){
			carry+=(ull)((ull)(F[i].real()+0.5)+(ull)(F[i].imag()+0.5)*FFT_BASE);
			num[i]=carry%BASE,carry/=BASE;
		}
		while(len>1&&!num[len-1]) len--;
		return *this;
#else
		int k=1,Len=2,n=len,m=b.len;
		while((1<<k)<n+m) k++,Len<<=1;
		Len<<=1,k++;
		if(Len>LENGTH) throw ERROR::MLE("FFT Length");
		Transform::FFT FFT_a,FFT_b;FFT_a.init(Len+1),FFT_b.init(Len+1);
		for(int i=0;i<len;i++) FFT_a.fft_a[i<<1]={(lf)(num[i]%FFT_BASE),0.0},FFT_a.fft_a[i<<1|1]={(lf)(num[i]/FFT_BASE),0.0};
		for(int i=0;i<b.len;i++) FFT_b.fft_a[i<<1]={(lf)(b.num[i]%FFT_BASE),0.0},FFT_b.fft_a[i<<1|1]={(lf)(b.num[i]/FFT_BASE),0.0};
		FFT_a.Init(k),FFT_b.Init(k);
		FFT_a.fft(1,Len);
		FFT_b.fft(1,Len);
		for(int i=0;i<Len;i++) FFT_a.fft_a[i]*=FFT_b.fft_a[i];
		FFT_a.fft(-1,Len);
		Expand(Len);
		std::fill(num.get(),num.get()+len,0);
		for(int i=0;i<Len;i+=2){
			int idx=i>>1;
			__uint128_t t=(ull)(FFT_a.fft_a[i|1].real()+0.5)*FFT_BASE+(ull)(FFT_a.fft_a[i].real()+0.5);
			num[idx]+=t%BASE;
			num[idx+1]+=num[idx]/BASE;
			num[idx]%=BASE;
			num[idx+1]+=t/BASE;
		}
		len=(Len>>1)+1;
		while(len>1&&!num[len-1]) len--;
		return *this;
#endif
	}
	UMultiInt& operator/=(const UMultiInt&b){*this=Mod(b).first;return *this;}
	UMultiInt operator/(const UMultiInt&b)const{return Mod(b).first;}
	UMultiInt& operator%=(const UMultiInt&b){*this=Mod(b).second;return *this;}
	UMultiInt operator%(const UMultiInt&b)const{return Mod(b).second;}
	UMultiInt& operator*=(const ull&b){
		if(b<=MAX){
			Expand(len+3),len+=3;
			for(int i=0;i<len;i++){num[i]*=b;}
			for(int i=0;i<len;i++){num[i+1]+=num[i]/BASE,num[i]%=BASE;}
			while(len>1&&!num[len-1]) len--;
		}
		else Mul(UMultiInt(b));
		return *this;
	}
	UMultiInt operator*(const ull&b){
		UMultiInt c(*this);c*=b;
		return c;
	}
	UMultiInt& operator/=(const ull&b){
		if(b==0) throw ERROR::Div_by_zero();
		int128 d=0;
		for(int i=len-1;i>=0;i--){
			d=d*BASE+num[i];num[i]=d/b;d%=b;
		}
		while(len>1&&!num[len-1]) len--;
		return *this;
	}
	UMultiInt operator/(const ull&b)const{
		UMultiInt c(*this);c/=b;
		return c;
	}
	UMultiInt& operator%=(const ull&b){
		if(b==0) throw ERROR::Div_by_zero();
		int128 d=0;
		for(int i=len-1;i>=0;i--){d=d*BASE+num[i];d%=b;}
		return *this=d;
	}
	UMultiInt operator%(const ull&b)const{UMultiInt res(*this);res%=b;return res;}
	UMultiInt& operator<<=(const ull&b){
		UMultiInt base("2");ull p=b;
		for(;p;p>>=1,base.square()) if(p&1) *this*=base;
		return *this;
	}
	UMultiInt& operator>>=(const ull&b){
		if(b<=10000){
			ull x=b;
			auto Div=[&](int cnt){
				ull d=0;
				for(int i=len-1;i>=0;i--){d=d*BASE+num[i];num[i]=(d>>cnt);d&=((1ull<<cnt)-1);}
				while(len>1&&!num[len-1]) len--;
				x-=cnt;
			};
			while(x>=LOG) Div(LOG);
			Div(x);
		}
		else *this/=UMultiInt("2").pow(b);
		return *this;
	}
	UMultiInt operator>>(const ull&b)const{UMultiInt res(*this);res>>=b;return res;}
	UMultiInt operator<<(const ull&b)const{UMultiInt res(*this);res<<=b;return res;}
	UMultiInt pow(const ull&b)const{
		if(b==0) return UMultiInt("1");
		if(b==1) return *this;
		UMultiInt res("1"),t(*this);ull p=b;
		for(;p;p>>=1){
			if(p&1) res*=t;
			if(p>1) t.square();
		}
		return res;
	}
	UMultiInt root(int m)const{
		if(m<0) throw ERROR::Negative("Index");
		if(is_zero()) return *this;
		if(m==1) return *this;
		UMultiInt x(std::min(*this,UMultiInt(BASE-1).Left((len+m-1)/m-1))),xx;
		int top=x.len-1;
		int l=0,r=BASE-1;
		while(l<r){
			int mid=(l+r)>>1;
			x.num[top]=mid;
			if(x.pow(m)<=*this) l=mid+1;
			else r=mid;
		}
		x.num[top]=l;
		while(x.len>1&&!x.num[x.len-1]) x.len--;
		xx=(x*(m-1)+*this/x.pow(m-1))/m;
		while(xx<x){
			std::swap(x,xx);
			xx=(x*(m-1)+*this/x.pow(m-1))/m;
		}
		return x;
	}
	UMultiInt sqrt()const{
		if(is_zero()) return *this;
		if(len==1) return (ull)std::sqrt(num[0]);
		if(len==2) return (ull)std::sqrt(num[1]*BASE+num[0]);
		UMultiInt x,xx;x.Expand((len+1)>>1),x.len=(len+1)>>1;
		int top=x.len-1;ull res=Get_sqrt();
		x.num[top]=res;
		while(x.len>1&&!x.num[x.len-1]) x.len--;
		xx=(x+*this/x)/2;
		while(xx<x) std::swap(x,xx),xx=(x+*this/x)/2;
		return x;
	}
	bool True()const{return !is_zero();}
	operator std::string()const{
		std::string s="";s+=std::to_string(num[len-1]);
		for(int i=len-2;i>=0;i--){char buf[10];sprintf(buf,"%08llu",num[i]);s+=buf;}
		return s;
	}
	void square(){
		if(len<=(int)T){Mul(*this);return;}
#ifndef __AVX2__
		int k=1,Len=2,n=len;
		while((1<<k)<(n<<1)) k++,Len<<=1;
		Len<<=1,k++;
		if(Len>LENGTH) throw ERROR::MLE("FFT Length");
		Transform::FFT FFT_a;FFT_a.init(Len+1);
		for(int i=0;i<len;i++) FFT_a.fft_a[i<<1]={(lf)(num[i]%FFT_BASE),0.0},FFT_a.fft_a[i<<1|1]={(lf)(num[i]/FFT_BASE),0.0};
		FFT_a.Init(k);
		FFT_a.fft(1,Len);
		for(int i=0;i<Len;i++) FFT_a.fft_a[i]*=FFT_a.fft_a[i];
		FFT_a.fft(-1,Len);
		Expand(Len);
		std::fill(num.get(),num.get()+len,0);
		for(int i=0;i<Len;i+=2){
			int idx=i>>1;
			__uint128_t t=(ull)(FFT_a.fft_a[i|1].real()+0.5)*FFT_BASE+(ull)(FFT_a.fft_a[i].real()+0.5);
			num[idx]+=t%BASE;
			num[idx+1]+=num[idx]/BASE;
			num[idx]%=BASE;
			num[idx+1]+=t/BASE;
		}
		len=(Len>>1)+1;
		while(len>1&&!num[len-1]) len--;
#else
		int _len=len+len,Len=2<<std::__lg(_len-1);
		if(Len>LENGTH) throw ERROR::MLE("FFT Length");
		Transform::FFT H;
		std::vector<Transform::Complex>F(Len,{0.0,0.0});
		for(int i=0;i<len;i++) F[i]={(lf)(num[i]%FFT_BASE),(lf)(num[i]/FFT_BASE)};
		H.init(Len);H.dif(F),H.mul(F,F),H.dit(F);
		std::fill(num.get(),num.get()+len,0);
		Expand(_len);len=_len;
		ull carry=0;
		for(int i=0;i<_len;i++){
			carry+=(ull)((ull)(F[i].real()+0.5)+(ull)(F[i].imag()+0.5)*FFT_BASE);
			num[i]=carry%BASE,carry/=BASE;
		}
		for(;carry&&Max>len;num[len++]=carry%BASE,carry/=BASE){
			if(len==Max) Expand(Max<<1);
		}
		while(len>1&&!num[len-1]) len--;
#endif
	}
	UMultiInt Square()const{
		UMultiInt res(*this);
		res.square();return res;
	}
	
	typedef UMultiInt<> UnsignedBigInt;
	typedef UMultiInt<2,26,67108864,8192> UnsignedBigBin;
	
	std::pair<UnsignedBigBin,UnsignedBigBin> _to_bin(const UnsignedBigInt&a,int l,int r){
		if(l==r) return {UnsignedBigBin(a.num[l]),UnsignedBigBin(UnsignedBigInt::BASE)};
		int mid=(l+r)>>1;
		auto [lo,pl]=_to_bin(a,l,mid);
		auto [hi,pr]=_to_bin(a,mid+1,r);
		pr*=pl;
		return {lo+=(hi*=pl),pr};
	}
	std::pair<UnsignedBigInt,UnsignedBigInt> _to_dec(const UnsignedBigBin&a,int l,int r){
		if(l==r) return {UnsignedBigInt(a.num[l]),UnsignedBigInt(UnsignedBigBin::BASE)};
		int mid=(l+r)>>1;
		auto [lo,pl]=_to_dec(a,l,mid);
		auto [hi,pr]=_to_dec(a,mid+1,r);
		pr*=pl;
		return {lo+=(hi*=pl),pr};
	}
};

typedef UMultiInt<> UnsignedBigInt;
typedef UMultiInt<2,26,67108864,8192> UnsignedBigBin;

#undef ull
#undef u32
#undef lf
#undef ll
#undef LENGTH
#undef int128


