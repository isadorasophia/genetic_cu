#include "streamer.h"

// ---- BufferStreamer ----
//
bool BufferStreamer::
Write(const void *buf, size_t len)
{
    if (m_HasOverflowed) return false;

    if (m_BytesUsed + len > m_Length) {
        m_HasOverflowed = true;
        return false;
    }

    memcpy(m_Buffer, buf, len);
    m_Buffer += len;
    m_BytesUsed += len;

    return true;
}

bool BufferStreamer::
Read(void *buf, size_t len)
{
    if (m_HasOverflowed) return false;

    if (m_BytesUsed + len > m_Length) {
        m_HasOverflowed = true;
        return false;
    }

    memcpy(buf, m_Buffer, len);
    m_Buffer += len;
    m_BytesUsed += len;        

    return true;
}
