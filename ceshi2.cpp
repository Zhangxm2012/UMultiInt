#pragma GCC target("fma")
#include<bits/stdc++.h>
#include<immintrin.h>
#define LENGTH (1<<24)
#define debug(str,sth) std::cout<<str<<sth<<'\n';
#ifndef __AVX2__
#define __AVX2__
#endif 
using u32=unsigned;
using ull=unsigned long long;
using lf=double;
using ll=long long;
using u128=__uint128_t;

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
		Number(const std::string&s):Exception("Error:"+s+" is not a valid numeric string!"){}
		Number(const std::string&s,u32 R):Exception("Error:"+s+" is not a valid numeric string in the case where the base is "+std::to_string(R)+"!"){}
	};
	class Out_of_range:public Exception{
	public:
		Out_of_range():Exception("Error:Out of range!"){}
	};
	class Base_ERROR:public Exception{
	public:
		Base_ERROR(u32 R1,u32 R2):Exception("Error:This base is "+std::to_string(R1)+",but other base is "+std::to_string(R2)+'!'){}
	};
}

namespace IO{
#include<ctype.h>
#include<assert.h>
#ifdef __SMALL__
	constexpr int c_i(char c){return (isdigit(c)?c-'0':(islower(c)?c-'a'+10:(isupper(c)?c-'A'+36:-1)));}
	constexpr char i_c(int i){return ((0<=i&&i<=9)?i+'0':((10<=i&&i<=35)?(i-10)+'a':(i-36)+'A'));}
#else
	constexpr int c_i(char c){return (isdigit(c)?c-'0':(isupper(c)?c-'A'+10:(islower(c)?c-'a'+36:-1)));}
	constexpr char i_c(int i){return ((0<=i&&i<=9)?i+'0':((10<=i&&i<=35)?(i-10)+'A':(i-36)+'a'));}
#endif
	struct Input{
		u32 table[0x10000],R;
		constexpr Input(u32 Radix):table(),R(Radix){
			u32 up=(u32)(i_c(R));
			for(u32 i=48;i<=up;i++){
				for(u32 j=48;j<=up;j++){
					table[i<<8|j]=c_i(i)*R+c_i(j);
				}
			}
		}
		Input(const Input&b):R(b.R){std::memcpy(table,b.table,sizeof table);}
		u32 operator()(const char*val)const{
			return table[(static_cast<short>(val[0])<<8)|static_cast<short>(val[1])];
		}
	};
	struct Output{
		constexpr u32 Pow(u32 Ra,u32 len){
			u32 t=1;
			for(;len;len>>=1,Ra*=Ra) if(len&1) t*=Ra;
			return t;
		}
		u32 LEN,RADIX;u128 table[10000];
		constexpr Output(u32 L,u32 R):LEN(L),RADIX(R),table(){
			auto size=Pow(R,L);
			for(u32 i=0;i<size;i++){
				u32 num=i;u128 t=0;
				for(int pos=L-1;pos>=0;pos--){
					u32 digit=num%RADIX;
					t|=static_cast<u128>(i_c(digit))<<(8*(pos));
					num/=RADIX;
				}
				table[i]=t;
			}
		}
		const char* operator()(const u32 val)const{
			return reinterpret_cast<const char*>(&table[val]);
		}
	};
	thread_local Input I_10(10),I_16(16);
	thread_local Output O_10(4,10),O_16(3,16);
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
		fComplex operator*(const T&b){return {rez*b,imz*b};}
		fComplex& operator*=(const T&b){rez*=b,imz*=b;return *this;}
		fComplex operator+(const T&x)const{return rez+=x;}
		fComplex conj()const{return {rez,-imz};}
		fComplex operator-()const{return {-rez,-imz};}
		T real()const{return rez;}
		T imag()const{return imz;}
	};
	using Complex=fComplex<lf>;
	Complex calc(const Complex&a,const Complex&b){return Complex(a.real()*b.real()+a.imag()*b.imag(),a.real()*b.imag()+a.imag()*b.real());}
	
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
	Complex calc(const Complex&a,const Complex&b){return _mm_fmadd_pd(_mm_unpacklo_pd(a.val,a.val),b.val,_mm_unpackhi_pd(a.val,a.val)*_mm_permute_pd(b.val,1));}	
#endif
	struct FFT{
		const lf pi=3.141592653589793;
		const lf pi2=6.283185307179586;
		Complex*omega;int size;
		FFT():omega(new Complex[1]()),size(1){*omega=Complex(1.0,0.0);}
		~FFT()noexcept{delete[]omega;}
		void init(int Len){
			if(Len<=(int)size<<1) return;
			int k=std::__lg(Len-1);
			omega=new Complex[1<<k](),omega[0]={1.0,0.0};Len=1<<k;
			for(int i=1;i<Len;i<<=1) omega[i]=std::polar(1.0,pi/(i<<1));
			for(int i=1;i<Len;i++) if(i&(i-1)) omega[i]=omega[i&(-i)]*omega[i&(i-1)];
		}
		void dif(Complex*a,int len){
			for(int Len=len>>1,sp=len;Len;sp=Len,Len>>=1){
				for(Complex *i=a;i!=a+Len;i++){auto temp=*i;*i=temp+i[Len],i[Len]=temp-i[Len];}
				for(Complex *blk=a+sp,*o=omega+1;blk!=a+len;blk+=sp,o++){
					for(Complex *i=blk;i!=blk+Len;i++){auto t1=*i,t2=i[Len]*(*o);*i=t1+t2,i[Len]=t1-t2;}
				}
			}
		}
		void dit(Complex*a,int len){
			for(int Len=1,sp=2;Len!=len;Len=sp,sp<<=1){
				for(Complex *i=a;i!=a+Len;i++){auto temp=*i;*i=temp+i[Len],i[Len]=temp-i[Len];}
				for(Complex *blk=a+sp,*o=omega+1;blk!=a+len;blk+=sp,o++){
					for(Complex* i=blk;i!=blk+Len;i++){auto t1=*i,t2=i[Len];*i=t1+t2,i[Len]=(t1-t2)*(*o).conj();}
				}
			}
		}
		void mul(Complex*F,Complex*G,int len){
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
}

std::ostream& operator<<(std::ostream&out,const u128&x){
	std::string res;u128 num=x;
	while(num){res+=(char)('0'+num%10),num/=10;}
	std::reverse(res.begin(),res.end());
	out<<res;
	return out;
}

using std::cout;

class UMultiInt{
private:
	static const u32 D=1,T=64;
	u32 RADIX,LEN,BASE,FFT;
	u32 *num,len,Max;
protected:
	//--help function--//
	bool is_zero()const{return (len==1&&num[0]==0);};
	struct Group{u32 L=0,B=0,F=0;};
	u32 Pow(u32 p)const{u32 t=1,base=RADIX;for(;p;p>>=1,base*=base){if(p&1) t*=base;}return t;}
	u32 log1(u32 p)const{u32 cnt=0;do{cnt++,p/=RADIX;}while(p);return cnt;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	u32 Log1(UT p,u32 base)const{u32 cnt=0;while(p){cnt++,p/=base;}return cnt;}
	Group Auto(u32 R)const{Group res;res.L=((Log1(100000000u,R)-1)>>1)<<1,res.B=Pow(res.L),res.F=Pow(res.L>>1);return res;}
	void trim(){while(len>1&&!num[len-1]) len--;}
	bool is_Pow(u32 x){return !(x&(x-1));}
	UMultiInt bit_helper(const UMultiInt&a,const UMultiInt&b,const std::function<bool(bool,bool)>&op)const{
		std::string _a=a.To_base(2),_b=b.To_base(2),_res;
		u32 n=_a.size(),m=_b.size(),lim=std::max(n,m);_res.resize(lim);
		std::cerr<<_a<<' '<<_b<<'\n';
		for(u32 i=0;i<lim;i++){_res[lim-i-1]='0'+op((i<n?_a[i]-'0':0),(i<m?_b[i]-'0':0));}
		std::cerr<<'\n'<<_res<<'\n';
		return UMultiInt(2,_res).To_base(a.RADIX);
	}
	void Expand(u32 _len){
		if(Max>=_len){len=_len;return;}
		u32* _num=new u32[Max=_len]();std::memcpy(_num,num,sizeof(u32)*len);
		delete[] num;num=_num;len=_len;
	}
	ull Get_sqrt()const{
		if(len<2) throw ERROR::Out_of_range();
		u32 top=len-1;ull res=num[top]*1ull*BASE+num[top-1];
		return std::sqrt(res)+1;
	}
	//--I/O helper--//
	std::string To_Str(u32 p)const{
		auto size=log1(p);
		std::string res;res.resize(size);
		for(u32 i=0;p;p/=RADIX,i++) res[size-1-i]=IO::i_c(p%RADIX);
		return res;
	}
	void Str_Dec(const char*val,u32 size){
		if(size==0||val[0]=='-') throw ERROR::Number(val);
		if(!std::all_of(val,val+size,[](char c)->bool{return isdigit(c);})) throw ERROR::Number(val);
		u32* cur=num+len-1;
		switch(size&7){
			case 0:{cur++;break;}
			case 1:{*cur=*val&15;break;}
			case 2:{*cur=IO::I_10(val);break;}
			case 3:{*cur=(*val&15)*100+IO::I_10(val+1);break;}
			case 4:{*cur=IO::I_10(val)*100+IO::I_10(val+2);break;}
			case 5:{*cur=(*val&15)*10000+IO::I_10(val+1)*100+IO::I_10(val+3);break;}
			case 6:{*cur=IO::I_10(val)*10000+IO::I_10(val+2)*100+IO::I_10(val+4);break;}
			case 7:{*cur=(*val&15)*1000000+IO::I_10(val+1)*10000+IO::I_10(val+3)*100+IO::I_10(val+5);break;}
		}
		for(const char* pos=val+(size&7);cur!=num;*--cur=IO::I_10(pos)*1000000+IO::I_10(pos+2)*10000+IO::I_10(pos+4)*100+IO::I_10(pos+6),pos+=8);
		trim();
	}
	void Str_Hex(const char*val,u32 size){
		if(size==0||val[0]=='-') throw ERROR::Number(val);
		if(!std::all_of(val,val+size,[](char c)->bool{return IO::c_i(c)<16;})) throw ERROR::Number(val);
		u32* cur=num+len-1;
		switch(size%6){
			case 0:{cur++;break;}
			case 1:{*cur=IO::c_i(*val);break;}
			case 2:{*cur=IO::I_16(val);break;}
			case 3:{*cur=(IO::c_i(*val)<<8)+IO::I_16(val+1);break;}
			case 4:{*cur=(IO::I_16(val)<<8)+IO::I_16(val+2);break;}
			case 5:{*cur=(IO::c_i(*val)<<16)+(IO::I_16(val+1)<<8)+IO::I_16(val+3);break;}
		}
		for(const char* pos=val+(size%6);cur!=num;*--cur=(IO::I_16(pos)<<16)+(IO::I_16(pos+2)<<8)+IO::I_16(pos+4),pos+=6);
		trim();
	}
	void From_Str(const char*val,u32 size){
		if(RADIX==10){Str_Dec(val,size);return;}
		if(RADIX==16){Str_Hex(val,size);return;}
		if(size==0||val[0]=='-') throw ERROR::Number(val);
		if(len>=LENGTH) throw ERROR::MLE(val);
		for(u32 i=size-1,j=0;~i;i--,j++) num[j/LEN]+=IO::c_i(val[i])*Pow(j%LEN);
		trim();
	}
	std::string Dec_str()const{
		if(!(*this)) return "0";
		u32 *p=num+len-1;
		std::string res=To_Str(*p);
		for(res.reserve(len*LEN);p--!=num;res.append(IO::O_10(*p/FFT),LEN>>1),res.append(IO::O_10(*p%FFT),LEN>>1));
		return res;
	}
	std::string Hex_str()const{
		if(!(*this)) return "0";
		u32 *p=num+len-1;
		std::string res=To_Str(*p);
		for(res.reserve(len*LEN);p--!=num;res.append(IO::O_16(*p/FFT),LEN>>1),res.append(IO::O_16(*p%FFT),LEN>>1));
		return res;
	}
	//--div&mod helper--
	ull Get(const UMultiInt&a,u32 pos)const{
		return 1ull*RADIX*BASE*(pos+1>=a.len?0:a.num[pos+1])+1ull*RADIX*a.num[pos]+(pos?a.num[pos-1]:0)/(BASE/RADIX);
	}
	std::pair<UMultiInt,UMultiInt> Simple_Mod(const UMultiInt&b)const{
		if(b.is_zero()) throw ERROR::Div_by_zero();
		if(*this<b) return std::make_pair(UMultiInt(RADIX),*this);
		UMultiInt Q(RADIX),R(*this);Q.Expand(len-b.len+1);
		ull b_Get=Get(b,b.len-1)+1;
		for(int i=len-b.len;i>=0;i--){
			ull q=0;
			auto Sub=[&](){
				ll t=0;
				for(u32 j=0;j<b.len;j++){
					t=t-q*b.num[j]+R.num[i+j];
					R.num[i+j]=(ull)(t%BASE),t/=BASE;
					if(R.num[i+j]>=BASE) R.num[i+j]+=BASE,t--;
				}if(t) R.num[i+b.len]+=(ull)(t);
				Q.num[i]+=q;
			};
			while((q=Get(R,i+b.len-1)/b_Get)){Sub();}q=1;
			for(u32 j=b.len-1;~j;j--){if(R.num[j+i]!=b.num[j]&&(q=b.num[j]<R.num[i+j],true)) break;}
			if(q) Sub();
		}Q.trim(),R.trim();
		return std::make_pair(Q,R);
	}
	UMultiInt Left(u32 cnt)const{
		if(len+cnt>LENGTH) throw ERROR::MLE("Left Shift");
		if(cnt==0||is_zero()) return *this;
		UMultiInt res(RADIX);
		res.Expand(len+cnt);
		std::copy(num,num+len,res.num+cnt);res.trim();
		return res;
	}
	UMultiInt Right(u32 cnt)const{
		if(cnt>=len) return UMultiInt(RADIX);
		UMultiInt res(RADIX);
		res.Expand(len-cnt);
		std::copy(num+cnt,num+len,res.num);res.trim();
		return res;
	}
	UMultiInt Inv(u32 n)const{
		if(is_zero()) throw ERROR::Div_by_zero();
		if(len<=T||n<=T+len){
			UMultiInt a(RADIX);a.Expand(n+1);
			std::fill(a.num,a.num+a.len,0);a.num[n]=1;
			return a.Simple_Mod(*this).first;
		}
		u32 k=(n-len+5)>>1,kk=k>len?0:len-k;
		UMultiInt t=Right(kk);
		u32 n1=k+t.len;
		UMultiInt t1=t.Inv(n1);
		UMultiInt res=(t1+t1).Left(n-n1-kk)-((*this)*t1*t1).Right(2*(n1+kk)-n);
		return --res;
	}
	std::pair<UMultiInt,UMultiInt> Mod(const UMultiInt&b)const{
		if(b.is_zero()) throw ERROR::Div_by_zero();
		if(*this<b) return std::make_pair(UMultiInt(RADIX),*this);
		if(len<=T||b.len<=T) return Simple_Mod(b);
		u32 Len=len-b.len+5,cnt=Len>b.len?0:b.len-Len;
		UMultiInt tem=b.Right(cnt);
		if(cnt) tem++;
		u32 inv=Len+tem.len;
		UMultiInt Q=(*this*tem.Inv(inv)).Right(inv+cnt);
		while(Q*b>*this) Q--;
		UMultiInt R=*this-Q*b;
		while(R>=b) Q++,R-=b;
		return std::make_pair(Q,R);
	}
public:
#define INIT(R) RADIX=R;auto helper=Auto(R);LEN=helper.L,BASE=helper.B,FFT=helper.F
//--construct--
	explicit UMultiInt(u32 R):num(new u32[D]()),len(1),Max(D){INIT(R);}
	UMultiInt(u32 R,const std::string&val){
		if(val.empty()) throw ERROR::Number(val);
		if(val[0]=='-') throw ERROR::Negative(val);
		if(!std::all_of(val.begin(),val.end(),[&](char c)->bool{return IO::c_i(c)!=-1&&IO::c_i(c)<(int)R;})) throw ERROR::Number(val,R);
		INIT(R);
		auto size=val.size();num=new u32[Max=len=(size+LEN-1)/LEN]();From_Str(val.data(),size);
	}
	UMultiInt(u32 R,const char*val){
		auto size=strlen(val);
		if(!size) throw ERROR::Number(val);
		if(val[0]=='-') throw ERROR::Negative(val);
		if(!std::all_of(val,val+size,[&](char c)->bool{return IO::c_i(c)!=-1&&IO::c_i(c)<(int)R;})) throw ERROR::Number(val,R);
		INIT(R);num=new u32[Max=len=(size+LEN-1)/LEN]();From_Str(val,size);
	}
	UMultiInt(const UMultiInt&b):RADIX(b.RADIX),LEN(b.LEN),BASE(b.BASE),FFT(b.FFT),num(reinterpret_cast<u32*>(std::memcpy(new u32[b.len],b.num,sizeof(u32)*b.len))),len(b.len),Max(b.len){}
	UMultiInt(UMultiInt&&b):RADIX(b.RADIX),LEN(b.LEN),BASE(b.BASE),FFT(b.FFT),num(b.num),len(b.len),Max(b.Max){b.num=new u32[b.Max=D];b.len=1;}
	~UMultiInt()noexcept{delete[] num;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt(u32 R,const UT&val){
		INIT(R);
		auto size=Log1(val,BASE);
		num=new u32[Max=size]();
		UT _val=val;len=0;
		do{num[len]=_val%BASE,len++,_val/=BASE;}while(_val);
	}
//--Assignment--
	UMultiInt& operator=(const UMultiInt&b){
		if(&b==this) return *this;
		RADIX=b.RADIX,LEN=b.LEN,BASE=b.BASE,FFT=b.FFT;
		if(Max<b.len) delete[] num,num=new u32[Max=b.len]();
		std::memcpy(num,b.num,sizeof(u32)*(len=b.len));
		return *this;
	}
	UMultiInt& operator=(UMultiInt&&b)noexcept{
		if(&b==this) return *this;
		RADIX=b.RADIX,LEN=b.LEN,BASE=b.BASE,FFT=b.FFT;
		delete[] num;num=b.num;len=b.len;Max=b.Max;
		b.num=new u32[b.Max=D]();b.len=1;
		return *this;
	}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt& operator=(const UT&val){
		auto size=Log1(val,BASE);
		if(size>Max){delete[] num;num=new u32[Max=size]();}
		UT _val=val;len=0;
		do{num[len]=_val%BASE,len++,_val/=BASE;}while(_val);
		return *this;
	}
//--To--
	operator std::string()const{
		if(RADIX==10) return Dec_str();
		if(RADIX==16) return Hex_str();
		if(!(*this)) return "0";
		u32 *p=num+len-1;
		std::string res=To_Str(*p);
		while(p--!=num){
			u32 digit=*p,cnt=LEN-log1(digit);
			for(u32 i=1;i<=cnt;i++) res+='0';
			res+=To_Str(digit);
		}
		return res;
	}
	operator bool()const noexcept{return !(len==1&&num[0]==0);}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	operator UT()const{
		UT res=0;
		for(u32 i=0;i<len;i++){res=res*BASE+num[i];}
		return res;
	}
	template<typename ST,typename std::enable_if<std::is_signed<ST>::value&&!std::is_floating_point<ST>::value>::type* =nullptr>
	operator ST()const{
		using UT=typename std::make_unsigned<ST>::type;
		return static_cast<ST>(UT(*this));
	}
//--function--
	int Two(){
		if(num[0]&1){return 0;}
		*this/=2u;
		return Two()+1;
	}
	size_t length()const{return len;}
	void check()const{
		std::cout<<"Radix:"<<RADIX<<'\n'<<"Len:"<<LEN<<'\n'<<"Base:"<<BASE<<'\n'<<"FFT_BASE:"<<FFT<<'\n';
		std::cout<<"len:"<<len<<'\n'<<"Max:"<<Max<<'\n';
		for(u32 i=0;i<len;i++) std::cout<<"num["<<i<<"]:"<<num[i]<<'\n';
	}
	u32 radix()const{return RADIX;}
	void test(u32 n){cout<<Inv(n)<<'\n';}
//--I/O--
	friend std::istream& operator>>(std::istream&in,UMultiInt&x){std::string buf;in>>buf;x=UMultiInt(x.RADIX,buf);return in;}
	friend std::ostream& operator<<(std::ostream&out,const UMultiInt&x){out<<std::string(x);return out;}
//--Compare--	
	int Cmp(const UMultiInt&b)const{
		if(b.RADIX!=RADIX) throw ERROR::Base_ERROR(RADIX,b.RADIX);
		if(len!=b.len) return (len>b.len)?1:-1;
		for(u32 i=len-1;~i;i--) if(num[i]!=b.num[i]) return (num[i]>b.num[i]?1:-1);
		return 0;
	}
#if __cplusplus>201703L
	std::strong_ordering operator<=>(const UMultiInt&b)const{return Cmp(b)<=>0;}
#endif
	bool operator<(const UMultiInt&b)const{return Cmp(b)<0;}
	bool operator<=(const UMultiInt&b)const{return Cmp(b)<=0;}
	bool operator>(const UMultiInt&b)const{return Cmp(b)>0;}
	bool operator>=(const UMultiInt&b)const{return Cmp(b)>=0;}
	bool operator==(const UMultiInt&b)const{return Cmp(b)==0;}
	bool operator!=(const UMultiInt&b)const{return Cmp(b)!=0;}
//--Operator--
	//--add--
	UMultiInt& operator+=(const UMultiInt&b){
		if(b.RADIX!=RADIX) throw ERROR::Base_ERROR(RADIX,b.RADIX);
		u32 n=std::max(len,b.len)+1;Expand(n+1);
		for(u32 i=0;i<n;i++) num[i]+=((i<b.len)?b.num[i]:0);
		for(u32 i=0;i<n;i++) if(num[i]>=BASE) num[i+1]+=num[i]/BASE,num[i]%=BASE;
		trim();
		return *this;
	}
	UMultiInt operator+(const UMultiInt&b){UMultiInt res(*this);res+=b;return res;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt operator+(const UT&b)const{UMultiInt res(*this);res+=b;return res;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt& operator+=(const UT&b){
		auto size=Log1(b,BASE);Expand(std::max(size,len)+1);
		UT digit=b;auto p=num;
		while(digit) *p+=digit%BASE,digit/=BASE,p++;
		for(u32 i=0;i<len;i++) if(num[i]>=BASE) num[i+1]+=num[i]/BASE,num[i]%=BASE;
		trim();return *this;
	}
	UMultiInt& operator++(){return *this+=1u;}
	UMultiInt operator++(int){UMultiInt res(*this);return *this+=1u,res;}
	//--sub--
	UMultiInt operator-(const UMultiInt&b)const{UMultiInt res(*this);res-=b;return res;}
	UMultiInt& operator-=(const UMultiInt&b){
		if(b.RADIX!=RADIX) throw ERROR::Base_ERROR(RADIX,b.RADIX);
		if(*this<b) throw ERROR::Negative("Result("+std::string(*this)+'-'+std::string(b)+')');
		ull t=0;
		for(u32 i=0;i<len;i++){
			ull sub=((i<b.len)?b.num[i]:0)+t;
			if(num[i]<sub) num[i]=BASE+num[i]-sub,t=1;
			else num[i]-=sub,t=0;
			if(t==0&&i>=b.len) break;
		}
		trim();return *this;
	}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt operator-(const UT&b)const{UMultiInt res(*this);res-=b;return res;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt& operator-=(const UT&b){
		auto size=Log1(b,BASE);
		if(len<=size) if(this->operator UT()<b) {
			throw ERROR::Negative("Result("+std::string(*this)+'-'+std::to_string(b)+')');
		}
		ull t=0;UT _b=b;
		for(u32 i=0;i<len;i++){
			ull sub=_b/BASE+t;
			if(num[i]<sub) num[i]=BASE+num[i]-sub,t=1;
			else num[i]-=sub,t=0;
			if(t==0&&i>=size) break;
		}
		trim();return *this;
	}
	UMultiInt& operator--(){return *this-=1u;}
	UMultiInt operator--(int){UMultiInt res(*this);return *this-=1u,res;}
	//--mul--
	UMultiInt operator*(const UMultiInt&b)const{UMultiInt res(*this);return res*=b;}
	UMultiInt& operator*=(const UMultiInt&b){
		if(b.RADIX!=RADIX) throw ERROR::Base_ERROR(RADIX,b.RADIX);
		if(len<T||b.len<T){
			u32 _len=len+b.len+1;
			ull* temp=new ull[_len+1]();
			for(u32 i=0;i<len;i++){
				for(u32 j=0;j<b.len;j++){
					temp[i+j]+=(ull)num[i]*(ull)b.num[j];
					temp[i+j+1]+=temp[i+j]/BASE,temp[i+j]%=BASE;
				}
			}
			Expand(_len+1);
			for(u32 i=0;i<_len;i++) num[i]=temp[i];
			for(u32 i=0;i<_len;i++) num[i+1]+=num[i]/BASE,num[i]%=BASE;
			trim();delete[] temp;return *this;
		}
		u32 _len=len+b.len,Len=2<<std::__lg(_len-1);
		if(Len>LENGTH) throw ERROR::MLE("FFT Length");
		Transform::FFT H;H.init(Len);
		Transform::Complex *F=new Transform::Complex[Len](),*G=new Transform::Complex[Len]();
		for(u32 i=0;i<len;i++) F[i]={(lf)(num[i]%FFT),(lf)(num[i]/FFT)};
		for(u32 i=0;i<b.len;i++) G[i]={(lf)(b.num[i]%FFT),(lf)(b.num[i]/FFT)};
		H.dif(F,Len),H.dif(G,Len),H.mul(F,G,Len),H.dit(F,Len);
		std::fill(num,num+len,0);Expand(Len);
		ull carry=0;
		for(u32 i=0;i<Len;i++){
			carry+=(ull)((ull)(F[i].real()+0.5)+(ull)(F[i].imag()+0.5)*FFT);
			num[i]=carry%BASE,carry/=BASE;
		}trim();
		return *this;
	}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt operator*(const UT&b)const{UMultiInt res(*this);res*=b;return res;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt& operator*=(const UT&b){
		u32 _len=len;
		auto size=len+Log1(b,BASE)+1;Expand(size);
		if((u128)b<(u128)ULLONG_MAX){
			u128 carry=0;
			for(u32 i=0;i<_len;i++){
				carry+=1ull*num[i]*b;
				num[i]=carry%BASE;carry/=BASE;
			}
			while(carry) num[++_len]=carry%BASE,carry/=BASE;
			trim();return *this;
		}
		else *this*=UMultiInt(RADIX,b);
		return *this;
	}
	//--div&mod--
	UMultiInt operator/(const UMultiInt&b)const{return Mod(b).first;}
	UMultiInt& operator/=(const UMultiInt&b){*this=Mod(b).first;return *this;}
	UMultiInt operator%(const UMultiInt&b)const{return Mod(b).second;}
	UMultiInt& operator%=(const UMultiInt&b){*this=Mod(b).second;return *this;}
	std::pair<UMultiInt,UMultiInt> mod(const UMultiInt&b)const{return Mod(b);}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt operator/(const UT&b)const{UMultiInt res(*this);res/=b;return res;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UMultiInt& operator/=(const UT&b){
		if(b==0) throw ERROR::Div_by_zero();
		if((u128)b>(u128)ULLONG_MAX){*this/=UMultiInt(RADIX,b);return *this;}
		u128 d=0;
		for(u32 i=len-1;~i;i--) d=d*BASE+num[i],num[i]=d/b,d%=b;
		trim();return *this;
	}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UT operator%(const UT&b)const{UMultiInt res(*this);res%=b;return res;}
	template<typename UT,typename std::enable_if<std::is_unsigned<UT>::value>::type* =nullptr>
	UT& operator%=(const UT&b){
		if(b==0) throw ERROR::Div_by_zero();
		if((u128)b>(u128)ULLONG_MAX){*this%=UMultiInt(RADIX,b);return *this;}
		u128 d=0;
		for(u32 i=len-1;~i;i--) d=d*BASE+num[i],d%=b;
		return *this=UMultiInt(RADIX,d);
	}
	//--pow&log--
	void square(){
		u32 _len=len<<1,Len=2<<std::__lg(_len-1);
		if(Len>LENGTH) throw ERROR::MLE("FFT Length");
		Transform::FFT H;H.init(Len);
		Transform::Complex *F=new Transform::Complex[Len]();
		for(u32 i=0;i<len;i++) F[i]={(lf)(num[i]%FFT),(lf)(num[i]/FFT)};
		H.dif(F,Len),H.mul(F,F,Len),H.dit(F,Len);
		std::fill(num,num+len,0);Expand(Len);
		ull carry=0;
		for(u32 i=0;i<Len;i++){
			carry+=(ull)((ull)(F[i].real()+0.5)+(ull)(F[i].imag()+0.5)*FFT);
			num[i]=carry%BASE,carry/=BASE;
		}trim();
	}
	UMultiInt pow(const u32&b)const{
		if(b==0) return UMultiInt(RADIX,"1");
		if(b==1) return *this;
		UMultiInt res(RADIX,"1"),t(*this);u32 p=b;
		for(;p;p>>=1){
			if(p&1) res*=t;
			if(p>1) t.square();
		}
		return res;
	}
	UMultiInt pow(const u32&b,const UMultiInt&mod)const{
		if(b==0) return UMultiInt(RADIX,"1");
		if(b==1) return Mod(mod).second;
		UMultiInt res(RADIX,"1"),t((*this)%mod);u32 p=b;
		for(;p;p>>=1){
			if(p&1) res*=t,res%=mod;
			if(p>1) t.square(),t%=mod;
		}
		return res<mod?res:res%mod;
	}
	UMultiInt root(u32 m)const{
		if(is_zero()) return *this;
		if(m==1) return *this;
		UMultiInt x(std::min(*this,UMultiInt(RADIX,BASE-1).Left((len+m-1)/m-1))),xx(RADIX);
		u32 top=x.len-1;u32 l=0,r=BASE-1;
		while(l<r){
			u32 mid=(l+r)>>1;
			x.num[top]=mid;
			if(x.pow(m)<=*this) l=mid+1;
			else r=mid;
		}
		x.num[top]=l;x.trim();
		xx=(x*(m-1)+*this/x.pow(m-1))/m;
		while(xx<x) std::swap(x,xx),xx=(x*(m-1)+*this/x.pow(m-1))/m;
		return x;
	}
	UMultiInt sqrt()const{
		if(is_zero()) return *this;
		if(len==1) return UMultiInt(RADIX,(ull)std::sqrt(num[0]));
		if(len==2) return UMultiInt(RADIX,(ull)std::sqrt(num[1]*BASE+num[0]));
		UMultiInt x(RADIX),xx(RADIX);x.Expand((len+1)>>1);
		u32 top=x.len-1;ull res=Get_sqrt();
		x.num[top]=res;x.trim();
		xx=(x+*this/x)/2u;
		while(xx<x) std::swap(x,xx),xx=(x+*this/x)/2u;
		return x;
	}
	lf log(u32 base,u32 a)const{return log10(a)/log10(base);}
	u32 log_r()const{return (len-1)*LEN+log1(num[len-1])-1;}
	u32 f_log(u32 r)const{
		if(is_zero()) throw ERROR::Div_by_zero();
		u32 lg_r=log_r()/(log(RADIX,r));
		UMultiInt ap=UMultiInt(RADIX,r).pow(lg_r);
		while((ap*=r)<=*this) lg_r++;
		return lg_r;
	}
	u32 c_log(u32 r)const{
		u32 e=f_log(r);
		if(UMultiInt(RADIX,r).pow(e)==*this) return e;
		else return e+1;
	}
	//--base_Transform--
	std::pair<UMultiInt,UMultiInt> _to(u32 ra,int l,int r)const{
		if(l==r) return {UMultiInt(ra,num[l]),UMultiInt(ra,BASE)};
		int mid=(l+r)>>1;
		auto [lo,pl]=_to(ra,l,mid);
		auto [hi,pr]=_to(ra,mid+1,r);
		pr*=pl;
		return {lo+=(hi*=pl),pr};
	}
	std::pair<UMultiInt,UMultiInt> _from(const UMultiInt&a,int l,int r)const{
		if(l==r) return {UMultiInt(RADIX,a.num[l]),UMultiInt(RADIX,a.BASE)};
		int mid=(l+r)>>1;
		auto [lo,pl]=_from(a,l,mid);
		auto [hi,pr]=_from(a,mid+1,r);
		pr*=pl;
		return {lo+=(hi*=pl),pr};
	}
	UMultiInt To_base(u32 r)const{return _to(r,0,len-1).first;}
	UMultiInt From_base(const UMultiInt&a)const{return _from(a,0,a.len-1).first;}
	//--bit--
	UMultiInt operator<<(u32 p)const{UMultiInt res(*this);res<<=p;return res;}
	UMultiInt& operator<<=(u32 p){*this*=UMultiInt(RADIX,2u).pow(p);return *this;}
    UMultiInt operator>>(u32 p)const{UMultiInt res(*this);res>>=p;return res;}
	UMultiInt& operator>>=(u32 p){*this/=UMultiInt(RADIX,2u).pow(p);return *this;}
	UMultiInt& operator^=(const UMultiInt&b){return *this=bit_helper(*this,b,[](bool a,bool b)->bool{return a^b;});}
	UMultiInt operator^(const UMultiInt&b)const{return bit_helper(*this,b,[](bool a,bool b)->bool{return a^b;});}
	UMultiInt& operator|=(const UMultiInt&b){return *this=bit_helper(*this,b,[](bool a,bool b)->bool{return a|b;});}
	UMultiInt operator|(const UMultiInt&b)const{return bit_helper(*this,b,[](bool a,bool b)->bool{return a|b;});}
	UMultiInt& operator&=(const UMultiInt&b){return *this=bit_helper(*this,b,[](bool a,bool b)->bool{return a&b;});}
	UMultiInt operator&(const UMultiInt&b)const{return bit_helper(*this,b,[](bool a,bool b)->bool{return a&b;});}
};

namespace Operation{
	UMultiInt Random(u32 r,u32 len){
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::uniform_int_distribution<int>u0(1,r-1),u1(0,r-1);
		std::string s="";
		s.push_back(IO::i_c(u0(gen)));
		for(u32 i=1;i<len;i++) s.push_back(IO::i_c(u1(gen)));
		return UMultiInt(r,s);
	}
	UMultiInt Pow(const UMultiInt&a,u32 p){return a.pow(p);}
	UMultiInt Pow(const UMultiInt&a,u32 p,const UMultiInt&mod){return a.pow(p,mod);}
	UMultiInt Fact(u32 r,u32 st,u32 n){
		if(n<=16){
			UMultiInt res(r);
			for(u32 i=st;i<st+n;i++) res*=i;
			return res;
		}
		u32 mid=(n+1)>>1;
		return Fact(r,st,mid)*Fact(r,st+mid,n-mid);
	}
	UMultiInt Fact(u32 r,u32 n){return Fact(r,1,n);}
	UMultiInt Gcd(const UMultiInt&a,const UMultiInt&b){
		UMultiInt c(a),d(b);
		u32 p=std::min(c.Two(),d.Two());
		u32 cnt=0;
		while(true){
			int res=c.Cmp(d);
			if(res>0) c-=d,c.Two();
			else if(res<0) d-=c,d.Two();
			else break;
			cnt++;
		}
		c<<=p;return c;
	}
	UMultiInt Root(const UMultiInt&a,u32 p){return a.root(p);}
	UMultiInt Sqrt(const UMultiInt&a){return a.sqrt();}
}

using namespace std;
using namespace Operation;
void Solve(){
	UMultiInt a(10),b(10);cin>>a>>b;
	cout<<(a^b)<<'\n';
}
int main(){
	// ios::sync_with_stdio(0);
	// cin.tie(0);
	// cout.tie(0);
	// freopen("in.in","r",stdin);
	// freopen("ceshi.out","w",stdout);
	int T=1;//cin>>T;
	while(T--) Solve();
	system("pause");
	return 0;
}
/*
input:
100001100010010011100100101100101010101010101010000101110010011
output:1000011000010111011101111101111111111
working:+ - * Simple_Mod
un: Inv / % 
Death Reason: 
	1.UMultiInt(&) operator*=
	2.UMultiInt(const UMultiInt&b)...[fake:Max(b.Max),real:Max(b.len)]
*/