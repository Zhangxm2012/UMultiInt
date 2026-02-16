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
		Out_of_range():Exception("Error:IO of range!"){}
	};
}

namespace IO{
#include<ctype.h>
#include<assert.h>
	constexpr int c_i(char c){return (isdigit(c)?c-'0':(isupper(c)?c-'A'+10:(islower(c)?c-'a'+36:-1)));}
	constexpr char i_c(int i){return ((0<=i&&i<=9)?i+'0':((10<=i&&i<=35)?(i-10)+'A':(i-36)+'a'));}
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
}

std::ostream& operator<<(std::ostream&out,const u128&x){
	std::string res;u128 num=x;
	while(num){res+=(char)('0'+num%10),num/=10;}
	std::reverse(res.begin(),res.end());
	out<<res;
	return out;
}

class UMultiInt{
#define INIT {RADIX=R;auto helper=Auto(R);LEN=helper.L,BASE=helper.B,FFT=helper.F;}
private:
	static const u32 D=4,T=256;
	u32 RADIX,LEN,BASE,FFT;
	u32 *num,len,Max;
	IO::Input I;
protected:
	struct Group{u32 L=0,B=0,F=0;};
	u32 Pow(u32 p)const{u32 t=1,base=RADIX;for(;p;p>>=1,base*=base){if(p&1) t*=base;}return t;}
	u32 log1(u32 p)const{u32 cnt=0;do{cnt++,p/=RADIX;}while(p);return cnt;}
	u32 Log1(ull p,u32 base)const{u32 cnt=0;while(p){cnt++,p/=base;}return cnt;}
	Group Auto(u32 R)const{Group res;res.L=(log1(1e8)>>1)<<1,res.B=Pow(res.L),res.F=Pow(res.L>>1);return res;}
	void trim(){while(len>1&&!num[len-1]) len--;}
	void Expand(u32 _len){
		if(Max>=_len) return;
		u32* _num=new u32[Max=_len]();std::memcpy(_num,num,sizeof(u32)*len);
		delete[] num;num=_num;len=_len;
	}
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
			case 2:{*cur=I(val);break;}
			case 3:{*cur=(*val&15)*100+I(val+1);break;}
			case 4:{*cur=I(val)*100+I(val+2);break;}
			case 5:{*cur=(*val&15)*10000+I(val+1)*100+I(val+3);break;}
			case 6:{*cur=I(val)*10000+I(val+2)*100+I(val+4);break;}
			case 7:{*cur=(*val&15)*1000000+I(val+1)*10000+I(val+3)*100+I(val+5);break;}
		}
		for(const char* pos=val+(size&7);cur!=num;*--cur=I(pos)*1000000+I(pos+2)*10000+I(pos+4)*100+I(pos+6),pos+=8);
		trim();
	}
	void Str_Hex(const char*val,u32 size){
		if(size==0||val[0]=='-') throw ERROR::Number(val);
		if(!std::all_of(val,val+size,[](char c)->bool{return IO::c_i(c)<16;})) throw ERROR::Number(val);
		u32* cur=num+len-1;
		switch(size%6){
			case 0:{cur++;break;}
			case 1:{*cur=IO::c_i(*val);break;}
			case 2:{*cur=I(val);break;}
			case 3:{*cur=(IO::c_i(*val)<<8)+I(val+1);break;}
			case 4:{*cur=(I(val)<<8)+I(val+2);break;}
			case 5:{*cur=(IO::c_i(*val)<<16)+(I(val+1)<<8)+I(val+3);break;}
		}
		for(const char* pos=val+(size%6);cur!=num;*--cur=(I(pos)<<16)+(I(pos+2)<<8)+I(pos+4),pos+=6);
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
public:
	UMultiInt(u32 R):num(new u32[D]()),len(1),Max(D),I(R){INIT;}
	UMultiInt(u32 R,const std::string&val):I(R){
		INIT;
		if(!std::all_of(val.begin(),val.end(),[&](char c)->bool{return IO::c_i(c)!=-1&&IO::c_i(c)<R;})) throw ERROR::Number(val,R);
		auto size=val.size();num=new u32[Max=len=(size+LEN-1)/LEN]();From_Str(val.data(),size);
	}
	UMultiInt(u32 R,const char*val):I(R){
		INIT;auto size=strlen(val);
		if(!std::all_of(val,val+size,[&](char c)->bool{return IO::c_i(c)!=-1&&IO::c_i(c)<R;})) throw ERROR::Number(val,R);
		num=new u32[Max=len=(size+LEN-1)/LEN]();From_Str(val,size);
	}
	UMultiInt(const UMultiInt&b):RADIX(b.RADIX),LEN(b.LEN),BASE(b.BASE),FFT(b.FFT),num(reinterpret_cast<u32*>(std::memcpy(new u32[b.len](),b.num,sizeof(u32)*b.len))),len(b.len),Max(b.Max),I(b.I){}
	UMultiInt(UMultiInt&&b):RADIX(b.RADIX),LEN(b.LEN),BASE(b.BASE),FFT(b.FFT),num(b.num),len(b.len),Max(b.Max),I(b.I){b.num=new u32[b.Max=D];b.len=1;}
	~UMultiInt()noexcept{delete[] num;}
	
	UMultiInt& operator=(const UMultiInt&b){
		if(&b==this) return *this;
		if(Max<b.len) delete[] num,num=new u32[len=Max=b.len]();
		std::memcpy(num,b.num,sizeof(u32)*(len=b.len));
		return *this;
	}
	UMultiInt& operator=(UMultiInt&&b)noexcept{
		if(this==&b){return *this;}
		delete[] num;num=b.num;len=b.len;Max=b.Max;
		b.num=new u32[b.Max=D]();b.len=1;
		return *this;
	}

	operator std::string()const{
		if(!(*this)) return "0";
		IO::Output O(LEN>>1,RADIX);
		u32 *p=num+len-1;
		std::string res=To_Str(*p);
		for(res.reserve(len*LEN);p--!=num;res.append(O(*p/FFT),LEN>>1),res.append(O(*p%FFT),LEN>>1));
		return res;
	}
	operator bool()const noexcept{return !(len==1&&num[0]==0);}

	size_t length()const{return len;}
	void check(){
		std::cout<<"Radix:"<<RADIX<<'\n'<<"Len:"<<LEN<<'\n'<<"Base:"<<BASE<<'\n'<<"FFT_BASE:"<<FFT<<'\n';
		std::cout<<"len:"<<len<<'\n'<<"Max:"<<Max<<'\n';
		for(u32 i=0;i<len;i++) std::cout<<"num["<<i<<"]:"<<num[i]<<'\n';
	}

	friend std::istream& operator>>(std::istream&in,UMultiInt&x){std::string buf;in>>buf;x=UMultiInt(x.RADIX,buf);return in;}
	friend std::ostream& operator<<(std::ostream&out,const UMultiInt&x){out<<std::string(x);return out;}
};
using namespace std;
int main(){
	// UMultiInt x(2,"1:11100101001");
	// cout<<x;
}
// 10000110001 00100111001001011001010101 01010101010000101110010011
/*
input:
100001100010010011100100101100101010101010101010000101110010011
output:1000011000010111011101111101111111111
*/
