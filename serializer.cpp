
/*!
  \file Serializer.cpp
  \author ggreco
  \brief contains class definitions of the serialization system
*/
/*****************************************************************************/

#include "serializer.h"
#include "streamer.h"
#include "endian.h"

//---------------------------------------------------------------------------//
  void SerialSizer::IO(uint8_t& p_Value) 
  {
    m_Length++;
  }
//---------------------------------------------------------------------------//
  void SerialSizer::IO(bool& p_Value) 
  {
    m_Length++;
  }
//---------------------------------------------------------------------------//
  void SerialSizer::IO(uint32_t& p_Value) 
  {
    m_Length+=4;
  }
//---------------------------------------------------------------------------//
  void SerialSizer::IO(std::string & p_Value) 
  {
    uint32_t dummy = p_Value.length();
    IO(dummy); 
    m_Length+=p_Value.length();
  }
//---------------------------------------------------------------------------//
  void SerialSizer::IO(uint64_t& p_Value) 
  {
    m_Length+=8;
  }
//---------------------------------------------------------------------------//
  void SerialSizer::IO(unsigned short& p_Value) 
  {
    m_Length+=2;
  }


//---------------------------------------------------------------------------//
  SerialSaver::SerialSaver(unsigned char* buf, size_t size) : m_internal(true) {
      m_streamer = new BufferStreamer(buf, size);
  } 
  SerialSaver::SerialSaver(std::ofstream &of) : m_internal(true) {
      m_streamer = new StreamStreamer(of);
  }
  SerialSaver::SerialSaver(FILE *f) : m_internal(true) {
      m_streamer = new PosixStreamer(f);
  }
  SerialSaver::~SerialSaver() {
    if (m_internal)
        delete m_streamer;
  }
//---------------------------------------------------------------------------//
  void SerialSaver::IO(uint8_t& p_Value) {
    m_streamer->Write(&p_Value, 1);
  }
//---------------------------------------------------------------------------//
  void SerialSaver::IO(uint32_t& p_Value) {
    uint32_t val = SwapLE16(p_Value);
    m_streamer->Write(&val, sizeof(uint32_t));
  }
//---------------------------------------------------------------------------//
  void SerialSaver::IO(uint16_t& p_Value) {
    uint16_t val = SwapLE16(p_Value);
    m_streamer->Write(&val, sizeof(uint16_t));
  }
//---------------------------------------------------------------------------//
  void SerialSaver::IO(std::string& p_Value) 
  {
    uint32_t lgth = p_Value.length();
    IO(lgth);
    if (lgth > 0)
        m_streamer->Write(p_Value.c_str(), lgth);
  }
  void SerialSaver::IO(uint64_t& p_Value)
  {
    uint64_t val = SwapLE64(p_Value); 
    m_streamer->Write(&val, sizeof(uint64_t));
  }
//---------------------------------------------------------------------------//
  void SerialSaver::IO(bool& p_Value)
  {
    uint8_t val = p_Value;
    IO(val);
  }
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
  SerialLoader::SerialLoader(unsigned char* buf, size_t size) : m_internal(true) {
       m_streamer = new BufferStreamer(buf, size);
  }
  SerialLoader::SerialLoader(std::ifstream &stream) : m_internal(true) {
       m_streamer = new StreamStreamer(stream);
  }
  SerialLoader::SerialLoader(FILE *file) : m_internal(true) {
       m_streamer = new PosixStreamer(file);
  }
  SerialLoader::~SerialLoader() { 
      if (m_internal)
    	  delete m_streamer;
  }
//---------------------------------------------------------------------------//
  void SerialLoader::IO(uint8_t & p_Value)
  {
    m_streamer->Read(&p_Value, 1);
  }
//---------------------------------------------------------------------------//
  void SerialLoader::IO(uint32_t& p_Value)
  {
    uint32_t val;

    if (!m_streamer->Read(&val, sizeof(uint32_t)))
        p_Value = 0;
    else
        p_Value = SwapLE32(val);
  }
//---------------------------------------------------------------------------//
  void SerialLoader::IO(std::string& p_Value) 
  {
    uint32_t l = 0;
    IO(l);
    if (l > 0) {
        char buffer[l];

        m_streamer->Read(buffer, l);
        p_Value.assign(buffer, l);
    }
    else
        p_Value.clear();
  }

  void SerialLoader::IO(uint64_t& p_Value)
  {
    uint64_t val;
    if (!m_streamer->Read((char *)&val, sizeof(uint64_t)))
        p_Value = 0;
    else
        p_Value = SwapLE64(val);
  }
//---------------------------------------------------------------------------//
  void SerialLoader::IO(uint16_t& p_Value)
  {
    uint16_t val;
    if (!m_streamer->Read((char *)&val, sizeof(uint16_t)))
        p_Value = 0;
    else
        p_Value = SwapLE16(val);
  }
//---------------------------------------------------------------------------//
  void SerialLoader::IO(bool& p_Value)
  {
    uint8_t val;
    IO(val);
    p_Value = val; 
  }
//---------------------------------------------------------------------------//

bool SerialLoader::
good() const
{ 
    return !m_streamer->Failed(); 
}
bool SerialSaver::
good() const
{ 
    return !m_streamer->Failed(); 
}


#ifdef TEST

#include <iostream>
#include <time.h>

struct point : public Serializable
{
    double x;
    short y;
    unsigned int z;
    time_t t;
    
    point() {}
    point(double a, short b, unsigned int c) : x(a), y(b), z(c) {}
    
    void serialize(Serializer &s) {
        s & x;
        s & y;
        s & z;
        s & t;
    }    
};

struct teststr : public Serializable
{
    int a;
    std::string b;
    bool c;
    std::map<int, point> d;
    
    void serialize(Serializer &s) {
        s & a;
        s & b;
        s & c;
        s & d;
    }
};

int i;
double d;
std::string mystr;
std::vector<teststr> myvec;
std::map<int, std::string> mymap;
int myarray[50];
std::string myarray2[40];

void serialize(Serializer &s)
{
    s & i;
    s & d;
    s & mystr;
    s & myvec;
    s & myarray;
    s & mymap;
    s & myarray2;
}

void saver()
{
   std::ofstream ofs("./PROVA");
    SerialSaver  saver(ofs);
    serialize(saver);
    ofs.close();
    
    std::cerr << "SerialSaver: " << saver.getBytesUsed() << "\n";
}

void loader()
{
    std::ifstream ifs("./PROVA");
    SerialLoader loader(ifs);
    serialize(loader);
//    std::cerr << "SerialLoader: " << loader.getBytesUsed() << "\n";
}

void dumper()
{
    std::cerr << "i = " << i 
              << "\nd = " << d
              << "\nmystr = " << mystr
              << "\nmyvec size = " << myvec.size() 
              << " myvec[2]={" << myvec[2].a << ',' << myvec[2].b << ',' << myvec[2].c << ",d[3]=(" << myvec[2].d[3].x << ',' << myvec[2].d[3].y << ',' << myvec[2].d[3].z <<")}"
              << " myvec[3]={" << myvec[3].a << ',' << myvec[3].b << ',' << myvec[3].c << ",d[3]=(" << myvec[3].d[3].x << ',' << myvec[3].d[3].y << ',' << myvec[3].d[3].z<< ")}"
              << "\nmyarray[0]=" << myarray[0] << " myarray[2]=" << myarray[2]
              << "\nmymap size = " << mymap.size()
              << " mymap[1]=" << mymap[1]
              << " mymap[5]=" << mymap[5]
              << "\nmyarray2[0]=" << myarray2[0] << " myarray2[1]=" << myarray2[1]
              << "\n";
}

void sizer()
{
    SerialSizer  test;
    serialize(test);
   
    std::cerr << "SerialSizer: " << test.getLength() << "\n";
    
}

void clearer()
{
    i = 0;
    d = 0.0;
    mystr.clear();
    myvec.clear();
    mymap.clear();
    for (int k = 0; k < 4; ++k)
        myarray2[k].clear();
}

void setter()
{
    i = -1234;
    d = 123.456;
    mystr = "Ciao, come stai?!?";
    for (int k = 0; k < 50; ++k)
        myarray[k] = 50 - k;

    for (int k = 0; k < 10; ++k) {
        char buffer[20];
        sprintf(buffer, "< mr.%d >", k + 1);
        std::string s(buffer);
        myarray2[0 + 4 *k] = s + "ciao";
        myarray2[1 + 4 *k] = s + "come" ;
        myarray2[2 + 4 *k] = s + "stai";
        myarray2[3 + 4 *k] = s + "tu?";
    }

    for (int k = 0; k < 150; ++k) {
        char ciao[10] = "ciaoXciao";
        struct teststr t;
        t.a = k * 2;
        ciao[4] = k + '0';
        ciao[5] = 0;
        t.b = ciao;
        t.c = ((k / 2) * 2 == k);
        t.d[k - 1] = point(k +2, k + 2, k + 2);
        t.d[k] = point(k +1, k + 2, k + 3);
        t.d[k + 1] = point(k +3, k + 2, k + 1);
        for (int v = 0; v < 20; ++v) 
            t.d[k - v] = point(k - v, k - v, k - v);

        myvec.push_back(t);
    }
    
    for (int k = 0; k < 150; ++k) {
        mymap[1 * k] = "uno";
        mymap[5 * k] = "cinque";
        mymap[20 *k] = "venti";
        mymap[10 *k] = "dieci";
    }
}

void saver_mem()
{
    unsigned char buffer[100000];
    SerialSaver ss(buffer, sizeof(buffer));
    serialize(ss);
    std::cerr << "Serializing from buffer to disk with " << ss.getBytesUsed() << " bytes\n";
    if (FILE *f = fopen("./PROVA", "wb")) {
        fwrite(buffer, 1, ss.getBytesUsed(), f);
        fclose(f);
    }
}

void loader_mem()
{
    unsigned char buffer[100000];
    if (FILE *f = fopen("./PROVA", "rb")) {
        int len = fread((char *)buffer, 1, sizeof(buffer), f);
        std::cerr << "Serializing from buffer to memory with " << len << " bytes\n";
        SerialLoader sl(buffer, len);
        serialize(sl);
        fclose(f);
    }
}

#include <sys/time.h>


timeval st, end;

inline void start_counter()  { gettimeofday(&st, NULL); }
inline void stop_counter() {
    gettimeofday(&end, NULL);
    std::cerr << "Time: " 
              << ((end.tv_sec - st.tv_sec) * 1000000 + (end.tv_usec - st.tv_usec))
              << " (usec)\n";
}

int main()
{
    setter();
    saver();
    clearer();
    start_counter();
    loader();
    stop_counter();
    dumper();
    clearer();
    start_counter();
    loader_mem();
    stop_counter();
    dumper();
}

#endif
