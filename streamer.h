#ifndef STREAMER_H

#define STREAMER_H

#include <iostream>

#include <cstdio>
#include <cstring>

class Streamer
{
public:
    Streamer() : m_failed(false) {}
    virtual ~Streamer() {}
    virtual bool Write(const void *, size_t) = 0;
    virtual bool Read(void *, size_t) = 0;
    bool Failed() const { return m_failed; }
protected:    
    bool m_failed;
};

class BufferStreamer : public Streamer
{
public:    
    BufferStreamer(unsigned char* buf, size_t size) :
        m_Buffer(buf), m_Length(size), m_BytesUsed(0),
        m_HasOverflowed(false) {}
    bool Write(const void *, size_t);
    bool Read(void *, size_t);

protected:
    /** @brief the actual buffer */
    unsigned char* m_Buffer;
    /** @brief the pointer pointing to the beginning of the buffer */
    unsigned char* m_BufferBeginning;
    /** @brief the total buffer length/size */
    size_t m_Length;
    /** @brief the number of bytes used within the buffer */
    unsigned long m_BytesUsed;
    /** @brief flag indicating whether the buffer as overflowed */
    bool m_HasOverflowed;

};

class PosixStreamer : public Streamer
{
public:    
    PosixStreamer(FILE *f) : m_file(f) {}
    bool Write(const void *v, size_t l) { 
        if (m_failed)
            return false;

        bool rc = ( fwrite(v, 1, l, m_file) == l); 

        m_failed = !rc;
        return rc;
    }
    virtual bool Read(void *v, size_t l) {
        if (m_failed)
            return false;
        
        bool rc = (fread(v, 1, l, m_file) == l);

        m_failed = !rc;
        return rc;
    }
protected:
    /** @brief the posix file descriptor */
    FILE *m_file;
};

class StreamStreamer : public Streamer
{
public:    
    StreamStreamer(std::ios &s) : m_stream(s) {}
    bool Write(const void *v, size_t l) { 
        if (m_failed)
            return false;

        bool rc = false;
        if (std::ostream *stream = dynamic_cast<std::ostream *>(&m_stream))        
            rc = stream->write((char *)v, l).good(); 

        m_failed = !rc;
        return rc;
    }
    virtual bool Read(void *v, size_t l) { 
        if (m_failed)
            return false;
        
        bool rc = false;
        if (std::istream *stream = dynamic_cast<std::istream *>(&m_stream))
            rc = stream->read((char *)v, l).good(); 

        m_failed = !rc;
        return rc;
    }
protected:
    /** @brief the file stream */
    std::ios &m_stream;
};

#endif
