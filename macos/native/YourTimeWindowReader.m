#import <ApplicationServices/ApplicationServices.h>
#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

// This process has no database or network code. It reads one foreground
// application's focused window and returns only a bounded title and hostname.
static NSString *readString(AXUIElementRef element, CFStringRef key) {
    CFTypeRef value = NULL;
    if (AXUIElementCopyAttributeValue(element, key, &value) != kAXErrorSuccess || !value) {
        return nil;
    }
    NSString *result = nil;
    if (CFGetTypeID(value) == CFStringGetTypeID()) {
        result = [(__bridge NSString *)value copy];
    }
    CFRelease(value);
    return result;
}

static NSString *safeHost(NSString *document) {
    if (!document || document.length > 8192) return nil;
    NSURLComponents *url = [NSURLComponents componentsWithString:document];
    NSString *scheme = url.scheme.lowercaseString;
    if (![scheme isEqualToString:@"https"] && ![scheme isEqualToString:@"http"]) return nil;
    NSString *host = url.host.lowercaseString;
    if (!host || host.length == 0 || host.length > 253) return nil;
    NSCharacterSet *allowed = [NSCharacterSet characterSetWithCharactersInString:@"abcdefghijklmnopqrstuvwxyz0123456789.-"];
    if ([host rangeOfCharacterFromSet:allowed.invertedSet].location != NSNotFound) return nil;
    return host;
}

static BOOL isBrowser(NSString *bundle) {
    NSString *name = bundle.lowercaseString;
    return [name containsString:@"chrome"] || [name containsString:@"safari"] ||
           [name containsString:@"firefox"] || [name containsString:@"edge"] ||
           [name containsString:@"arc"];
}

static BOOL isSensitiveApp(NSString *bundle) {
    NSString *name = bundle.lowercaseString;
    return [name containsString:@"password"] || [name containsString:@"keychain"] ||
           [name hasPrefix:@"com.1password"] || [name hasPrefix:@"com.agilebits"] ||
           [name containsString:@"bitwarden"] || [name containsString:@"keepassxc"] ||
           [name containsString:@"dashlane"];
}

static pid_t visibleProcessUnderFluid(void) {
    CFArrayRef windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID);
    if (!windows) return 0;
    pid_t selected = 0;
    for (NSDictionary *window in (__bridge NSArray *)windows) {
        if ([window[(NSString *)kCGWindowLayer] integerValue] != 0 ||
            [window[(NSString *)kCGWindowAlpha] doubleValue] <= 0) continue;
        CGRect bounds = CGRectZero;
        NSDictionary *shape = window[(NSString *)kCGWindowBounds];
        if (!shape || !CGRectMakeWithDictionaryRepresentation((__bridge CFDictionaryRef)shape, &bounds) ||
            bounds.size.width < 350 || bounds.size.height < 250) continue;
        pid_t candidate = [window[(NSString *)kCGWindowOwnerPID] intValue];
        if (candidate > 0) { selected = candidate; break; }
    }
    CFRelease(windows);
    return selected;
}

static void emit(NSDictionary *object) {
    NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
    if (data) {
        fwrite(data.bytes, 1, data.length, stdout);
        fputc('\n', stdout);
    }
}

static void saveSample(NSDictionary *object) {
    NSString *directory = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support/personal-activity-ledger"];
    NSDictionary *attributes = @{NSFilePosixPermissions: @0700};
    if (![[NSFileManager defaultManager] createDirectoryAtPath:directory
                                    withIntermediateDirectories:YES
                                                     attributes:attributes error:NULL]) return;
    NSString *path = [directory stringByAppendingPathComponent:@"window-reader-latest.json"];
    NSString *temporary = [directory stringByAppendingFormat:@"/.window-reader-%d.tmp", getpid()];
    NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
    if (!data) return;
    int fd = open(temporary.fileSystemRepresentation, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW, 0600);
    if (fd < 0) return;
    fchmod(fd, 0600);
    const uint8_t *bytes = data.bytes;
    size_t remaining = data.length;
    while (remaining) {
        ssize_t written = write(fd, bytes, remaining);
        if (written <= 0) break;
        bytes += written;
        remaining -= (size_t)written;
    }
    if (remaining == 0 && fsync(fd) == 0) {
        close(fd);
        if (rename(temporary.fileSystemRepresentation, path.fileSystemRepresentation) != 0)
            unlink(temporary.fileSystemRepresentation);
    } else {
        close(fd);
        unlink(temporary.fileSystemRepresentation);
    }
}

static void sampleFrontmost(void) {
    BOOL trusted = AXIsProcessTrusted();
    NSRunningApplication *front = NSWorkspace.sharedWorkspace.frontmostApplication;
    pid_t pid = front ? front.processIdentifier : 0;
    NSString *bundle = front.bundleIdentifier;
    if ([bundle isEqualToString:@"com.FluidApp.app"]) {
        pid_t underlying = visibleProcessUnderFluid();
        if (underlying > 0) {
            pid = underlying;
            bundle = [NSRunningApplication runningApplicationWithProcessIdentifier:pid].bundleIdentifier;
        }
    }
    CFDictionaryRef session = CGSessionCopyCurrentDictionary();
    NSDictionary *sessionInfo = (__bridge NSDictionary *)session;
    BOOL locked = !session || [sessionInfo[@"CGSSessionScreenIsLocked"] boolValue];
    if (session) CFRelease(session);
    double idle = CGEventSourceSecondsSinceLastEventType(kCGEventSourceStateCombinedSessionState,
                                                         kCGAnyInputEventType);
    NSString *title = nil;
    NSString *host = nil;
    if (trusted && !locked && idle < 180 && pid > 0 && !isSensitiveApp(bundle)) {
        AXUIElementRef app = AXUIElementCreateApplication(pid);
        if (app) {
            AXUIElementSetMessagingTimeout(app, 0.25f);
            CFTypeRef window = NULL;
            AXError status = AXUIElementCopyAttributeValue(app, kAXFocusedWindowAttribute, &window);
            CFRelease(app);
            if (status == kAXErrorSuccess && window && CFGetTypeID(window) == AXUIElementGetTypeID()) {
                AXUIElementSetMessagingTimeout((AXUIElementRef)window, 0.25f);
                title = readString((AXUIElementRef)window, kAXTitleAttribute);
                if (isBrowser(bundle))
                    host = safeHost(readString((AXUIElementRef)window, kAXDocumentAttribute));
            }
            if (window) CFRelease(window);
        }
    }
    if (title.length > 500) title = [title substringToIndex:500];
    NSISO8601DateFormatter *clock = [[NSISO8601DateFormatter alloc] init];
    clock.formatOptions = NSISO8601DateFormatWithInternetDateTime | NSISO8601DateFormatWithFractionalSeconds;
    saveSample(@{@"sampled_at_utc": [clock stringFromDate:[NSDate date]],
                 @"trusted": @(trusted), @"pid": @(pid),
                 @"title": title ?: [NSNull null], @"site_host": host ?: [NSNull null]});
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        BOOL trusted = AXIsProcessTrusted(); // Never prompts for a grant.
        if (argc == 2 && strcmp(argv[1], "--request-access") == 0) {
            // Explicit installation/recovery only; --watch never prompts.
            NSDictionary *options = @{(__bridge NSString *)kAXTrustedCheckOptionPrompt: @YES};
            trusted = AXIsProcessTrustedWithOptions((__bridge CFDictionaryRef)options);
            emit(@{@"trusted": @(trusted)});
            return 0;
        }
        if (argc == 2 && strcmp(argv[1], "--status") == 0) {
            emit(@{@"trusted": @(trusted)});
            return 0;
        }
        if (argc != 2 || strcmp(argv[1], "--watch") != 0) return 2;
        umask(077);
        while (1) {
            @autoreleasepool { sampleFrontmost(); }
            // NSWorkspace refreshes its foreground application through its
            // notification/run loop. sleep() left the initial PID cached for
            // hours while the app continued to report a fresh sample time.
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:5]];
        }
    }
}
