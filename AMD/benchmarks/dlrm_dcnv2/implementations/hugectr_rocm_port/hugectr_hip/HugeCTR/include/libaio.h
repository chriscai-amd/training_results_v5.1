// Phase 14p libaio.h shim (2026-05-14). The container we build in lacks
// libaio.h headers (libaio-dev unavailable in apt cache + no networking).
// This minimal shim declares the libaio APIs HugeCTR's aio_context.cpp uses;
// link still resolves against /usr/lib/.../libaio.so.1 from the host.
//
// Equivalent to the upstream libaio.h subset for our needs only.
#ifndef _LIBAIO_H
#define _LIBAIO_H 1

#include <sys/types.h>
#include <sys/uio.h>
#include <time.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>  // strerror — upstream libaio.h pulls this in transitively

#ifdef __cplusplus
extern "C" {
#endif

typedef struct io_context* io_context_t;

typedef enum io_iocb_cmd {
    IO_CMD_PREAD       = 0,
    IO_CMD_PWRITE      = 1,
    IO_CMD_FSYNC       = 2,
    IO_CMD_FDSYNC      = 3,
    IO_CMD_POLL        = 5,
    IO_CMD_NOOP        = 6,
    IO_CMD_PREADV      = 7,
    IO_CMD_PWRITEV     = 8,
} io_iocb_cmd_t;

struct io_iocb_common {
    void*   buf;
    uint64_t nbytes;
    int64_t  offset;
    int64_t  __pad3;
    uint32_t flags;
    uint32_t resfd;
};

struct iocb {
    void*    data;
    uint32_t key;
    uint32_t aio_rw_flags;
    short    aio_lio_opcode;
    short    aio_reqprio;
    int      aio_fildes;
    union {
        struct io_iocb_common  c;
        char __pad[40];  // ensure size matches kernel iocb (~64 bytes total)
    } u;
};

struct io_event {
    void*    data;
    struct iocb* obj;
    long long res;
    long long res2;
};

extern int io_queue_init(int maxevents, io_context_t* ctxp);
extern int io_queue_release(io_context_t ctx);
extern int io_submit(io_context_t ctx, long nr, struct iocb* ios[]);
extern int io_cancel(io_context_t ctx, struct iocb* iocb, struct io_event* evt);
extern int io_getevents(io_context_t ctx_id, long min_nr, long nr,
                        struct io_event* events, struct timespec* timeout);

static inline void io_prep_pread(struct iocb* iocb, int fd, void* buf,
                                 size_t count, long long offset) {
    iocb->data = 0;
    iocb->aio_fildes = fd;
    iocb->aio_lio_opcode = IO_CMD_PREAD;
    iocb->aio_reqprio = 0;
    iocb->u.c.buf = buf;
    iocb->u.c.nbytes = count;
    iocb->u.c.offset = offset;
}

#ifdef __cplusplus
}
#endif

#endif
