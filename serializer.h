#ifndef SERIALIZER_H

#include <list>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <netinet/in.h>

#define SERIALIZER_H
  class Serializer;
  class Streamer;

  class Serializable
  {
  public:
    /// loads/stores all aspects of the object for persistence
    virtual void serialize(Serializer& p_Archive)=0;
    virtual ~Serializable() {};
  };

  class Serializer
  {
  public:

    /** @brief symbolic names for possible Serializer based types
     */
    enum eSerializerType
    {
      SizerType,
      SaverType,                   //!< the object is a SerialSaver object
      LoaderType                  //!< the object is a SerialLoader object
    };

    virtual bool good() const { return true; }

    virtual void operator&(Serializable& p_Serializable)
    {
      p_Serializable.serialize(*this);
    }
    /** @brief serialization operator for base type objects (for which IO() is defined) */
    // char family
    void operator&(unsigned char& p_Object) { IO(p_Object);  }
    void operator&(char& p_Object) { IO((uint8_t &)p_Object);  }

    // short family
    void operator&(unsigned short& p_Object) { IO(p_Object);  }
    void operator&(short& p_Object) { IO((uint16_t &)p_Object);  }
    
    // long family (32bit)
    void operator&(unsigned long& p_Object) { IO((uint32_t &)p_Object);  }
    void operator&(int& p_Object) { IO((uint32_t &)p_Object);  }
    void operator&(unsigned int& p_Object) { IO((uint32_t &)p_Object);  }
    void operator&(long& p_Object) { IO((uint32_t &)p_Object);  }
    void operator&(float& p_Object) { IO((uint32_t &)p_Object);  }

    // long long family (64bit)
    void operator&(double& p_Object) { IO((uint64_t &)p_Object);  }
    void operator&(long long& p_Object) { IO((uint64_t &)p_Object);  }
    void operator&(unsigned long long& p_Object) { IO((uint64_t &)p_Object);  }
    
    // special types
    void operator&(bool& p_Object) { IO(p_Object);  }
    void operator&(std::string& p_Object) { IO(p_Object);  }
    
    template <typename T, std::size_t N> void operator &(T(&obj)[N]) {
        for (size_t i = 0; i < N; ++i)
            *this & obj[i];
    }
    /** @brief generalized method for serializing a stl list */
    template<class T> void operator&(std::list<T>& p_ObjectList)
    {
      unsigned long  count = p_ObjectList.size();
      *this & count;
      if(getType() == Serializer::LoaderType)
      {
        p_ObjectList.clear();
        T obj;
        for(unsigned long  i=0;i<count;++i)
        {
          *this & obj;
          p_ObjectList.push_back(obj);
        }
      }
      else
      {
        T obj;
        for(typename std::list<T>::iterator it = p_ObjectList.begin();
          it != p_ObjectList.end(); ++it)
        {
          obj = *it;
          *this & obj;
        }
      }
    }
    /** @brief generalized method for serializing a stl vector */
    template<class T> void operator&(std::vector<T>& p_ObjectList)
    {
      unsigned long count = p_ObjectList.size();
      *this & count;
      if(getType() == Serializer::LoaderType)
      {
        p_ObjectList.clear();
        T obj;
        for(unsigned long  i=0;i<count;++i)
        {
          *this & obj;
          p_ObjectList.push_back(obj);
        }
      }
      else
      {
        T obj;
        for(typename std::vector<T>::iterator it = p_ObjectList.begin();
          it != p_ObjectList.end(); ++it)
        {
          obj = *it;
          *this & obj;
        }
      }
    }
    template<class T> void operator&(std::set<T>& p_ObjectList)
    {
      unsigned long count = p_ObjectList.size();
      *this & count;
      if(getType() == Serializer::LoaderType)
      {
        p_ObjectList.clear();
        T obj;
        for(unsigned long  i=0;i<count;++i)
        {
          *this & obj;
          p_ObjectList.insert(obj);
        }
      }
      else
      {
        T obj;
        for(typename std::set<T>::iterator it = p_ObjectList.begin();
          it != p_ObjectList.end(); ++it)
        {
          obj = *it;
          *this & obj;
        }
      }
    }
    /** @brief generalized method for serializing a stl map */
    template<class T, class U> void operator&(std::map<T,U>& p_ObjectMap)
    {
      unsigned long  count = p_ObjectMap.size();
      *this & count;
      if(getType() == Serializer::LoaderType)
      {
        p_ObjectMap.clear();
        T key;
        U value;
        for(unsigned long  i=0;i<count;++i)
        {
          *this & key;
          *this & value;
          p_ObjectMap[key] = value;
        }
      }
      else
      {
        T key;
        U value;
        for(typename std::map<T,U>::iterator it = p_ObjectMap.begin();
          it != p_ObjectMap.end(); ++it)
        {
          key   = it->first;
          value = it->second;
          *this & key;
          *this & value;
        }
      }
    }
    /** @brief generalized method for serializing a stl pair */
    template<class T, class U> void operator&(std::pair<T,U>& p_ObjectPair)
    {
      *this & p_ObjectPair.first;
      *this & p_ObjectPair.second;
    }

    /** @brief inputs/outputs a single UChar value
     *
     * @param p_Value the value to input/output
     */
    virtual void IO(uint8_t & p_Value)=0;
    /** @brief inputs/outputs a single UShort value
     *
     * @param p_Value the value to input/output
     */
    virtual void IO(uint16_t & p_Value)=0;

    /** @brief inputs/outputs a single ULong value
     *
     * @param p_Value the value to input/output
     */
    virtual void IO(uint32_t& p_Value)=0;

    /** @brief inputs/outputs a single String value
     *
     * @param p_Value the value to input/output
     */
    virtual void IO(std::string& p_Value)=0;

    /** @brief inputs/outputs a single Float value
     *
     * @param p_Value the value to input/output
     */
    virtual void IO(uint64_t& p_Value)=0;

    /** @brief inputs/outputs a single bool value
     *
     * @param p_Value the value to input/output
     */
    virtual void IO(bool& p_Value)=0;

    /** @brief queries the type of the object (symbolic name)
     */
    virtual int getType() const = 0;

  }; // end of Serializer class declaration

//----------------------------------------------------------------------------//

  /**
   * @brief handles serial sizing
   *
   */
  class SerialSizer : public Serializer
  {
  public:

    /** @brief default constructor
     *
    */
    inline SerialSizer()
      : m_Length( 0)
    {}

    // doc for the following gets copied
    virtual void IO(uint8_t& p_Value);
    virtual void IO(uint16_t& p_Value);
    virtual void IO(uint32_t& p_Value);
    virtual void IO(uint64_t& p_Value);
    virtual void IO(std::string& p_Value);
    virtual void IO(bool& p_Value);

    /** @brief Return the type of this Serializer
     */
    inline virtual int getType() const
    {
      return SizerType;
    }

    /** @brief returns the number of bytes inputted/outputted
     */
    inline unsigned long getLength() const
    {
      return m_Length;
    }

  protected:

    /** @brief the number of bytes streamed */
    size_t m_Length;

  }; // end of SerialSizer class declaration

//----------------------------------------------------------------------------//

  /**
   * @brief implements the Serializer interface for saving information
   */
  class SerialSaver : public Serializer
  {
  public:

    /** @brief constructor for memory saver
     *
     * @param p_Buffer the buffer to fill by this object
     * @param p_Size the size of the buffer to fill by this object
     */
    SerialSaver( unsigned char* p_Buffer, size_t p_Size);
    /** @brief constructor for C++ streams saver
     *
     * @param stream the output file stream to write to.
     */
    SerialSaver( std::ofstream &stream);
    /** @brief constructor for posix FILE saver
     *
     * @param file the posix FILE to write to.
     */
    SerialSaver( FILE *file);
    /** @brief constructor for network saver
     *
     * @param fd a TCP or connected UDP socket descriptor.
     */
    SerialSaver( int fd );
    SerialSaver(Streamer &s) : m_streamer(&s), m_internal(false) {}
    SerialSaver(Streamer *s) : m_streamer(s), m_internal(false) {}

    ~SerialSaver();

    // doc for the following gets copied
    virtual void IO(uint8_t& p_Value);
    virtual void IO(uint16_t& p_Value);
    virtual void IO(uint32_t& p_Value);
    virtual void IO(uint64_t& p_Value);
    virtual void IO(std::string& p_Value);
    virtual void IO(bool& p_Value);

    virtual bool good() const; 

    /** @brief Return the type of this Serializer
     */
    inline virtual int getType() const
    {
      return SaverType;
    }
   protected:
    Streamer *m_streamer;
    bool m_internal;

  }; // end of SerialSaver class declaration

//----------------------------------------------------------------------------//

  /**
   * @brief implements the Serializer interface for loading serialized information
   *
   */
  class SerialLoader : public Serializer
  {
  public:

    /** @brief constructor
     *
     * @param p_Buffer the buffer to read from
     * @param p_Size the size of the buffer to read from
     */
    SerialLoader( unsigned char *p_Buffer, size_t p_Size);
    SerialLoader( std::ifstream &stream);
    SerialLoader( FILE *file);
    SerialLoader( int fd);
    SerialLoader(Streamer &s) : m_streamer(&s), m_internal(false) {}
    SerialLoader(Streamer *s) : m_streamer(s), m_internal(false) {}

	~SerialLoader();

    // doc for the following gets copied
    virtual void IO(uint8_t& p_Value);
    virtual void IO(uint16_t& p_Value);
    virtual void IO(uint32_t& p_Value);
    virtual void IO(uint64_t& p_Value);
    virtual void IO(std::string& p_Value);
    virtual void IO(bool& p_Value);

    virtual bool good() const; 

    inline virtual int getType() const
    {
      return LoaderType;
    }

  protected:

	Streamer *m_streamer;
    bool m_internal;
  }; // end of SerialLoader class declaration



#endif
