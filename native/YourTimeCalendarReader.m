#import <Foundation/Foundation.h>
#import <EventKit/EventKit.h>
#import <sys/stat.h>

static EKEventStore *store;
static NSISO8601DateFormatter *formatter;

static NSString *statusFor(EKEntityType type) {
    EKAuthorizationStatus status = [EKEventStore authorizationStatusForEntityType:type];
    switch (status) {
        case EKAuthorizationStatusNotDetermined: return @"not_determined";
        case EKAuthorizationStatusRestricted: return @"restricted";
        case EKAuthorizationStatusDenied: return @"denied";
        case EKAuthorizationStatusFullAccess: return @"full_access";
        case EKAuthorizationStatusWriteOnly: return @"write_only";
        default: return @"unknown";
    }
}
static BOOL readable(EKEntityType type) {
    NSString *value = statusFor(type);
    return [value isEqualToString:@"full_access"] || [value isEqualToString:@"authorized"];
}
static void output(NSDictionary *value) {
    NSData *bytes = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingSortedKeys error:nil];
    if (bytes) {
        fwrite(bytes.bytes, 1, bytes.length, stdout);
        fwrite("\n", 1, 1, stdout);
    }
}
static void waitUntil(BOOL (^finished)(void), NSTimeInterval seconds) {
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:seconds];
    while (!finished() && [deadline timeIntervalSinceNow] > 0) {
        [[NSRunLoop currentRunLoop] runMode:NSDefaultRunLoopMode
                                beforeDate:[NSDate dateWithTimeIntervalSinceNow:0.2]];
    }
}
static NSDictionary *requestAccess(void) {
    __block BOOL eventsGranted = NO, eventDone = NO;
    [store requestFullAccessToEventsWithCompletion:^(BOOL granted, NSError *error) {
        eventsGranted = granted; eventDone = YES;
    }];
    waitUntil(^{ return eventDone; }, 120);
    __block BOOL remindersGranted = NO, reminderDone = NO;
    [store requestFullAccessToRemindersWithCompletion:^(BOOL granted, NSError *error) {
        remindersGranted = granted; reminderDone = YES;
    }];
    waitUntil(^{ return reminderDone; }, 120);
    return @{ @"eventsGranted": @(eventsGranted), @"remindersGranted": @(remindersGranted),
              @"eventsStatus": statusFor(EKEntityTypeEvent),
              @"remindersStatus": statusFor(EKEntityTypeReminder) };
}
static NSDictionary *listCalendars(void) {
    NSMutableArray *events = [NSMutableArray array], *reminders = [NSMutableArray array];
    if (readable(EKEntityTypeEvent)) {
        for (EKCalendar *calendar in [store calendarsForEntityType:EKEntityTypeEvent]) {
            [events addObject:@{ @"id": calendar.calendarIdentifier ?: @"", @"name": calendar.title ?: @"" }];
        }
    }
    if (readable(EKEntityTypeReminder)) {
        for (EKCalendar *calendar in [store calendarsForEntityType:EKEntityTypeReminder]) {
            [reminders addObject:@{ @"id": calendar.calendarIdentifier ?: @"", @"name": calendar.title ?: @"" }];
        }
    }
    return @{ @"eventsStatus": statusFor(EKEntityTypeEvent),
              @"remindersStatus": statusFor(EKEntityTypeReminder),
              @"eventCalendars": events, @"reminderLists": reminders };
}
static NSDictionary *scopeMap(NSArray *items) {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSDictionary *item in items) {
        NSString *identity = item[@"id"];
        if ([identity isKindOfClass:[NSString class]] && identity.length > 0)
            result[identity] = @([item[@"includeTitle"] boolValue]);
    }
    return result;
}
static NSDictionary *exportScope(NSString *scopePath, NSString *outputPath) {
    NSData *scopeData = [NSData dataWithContentsOfFile:scopePath];
    NSDictionary *scope = scopeData ? [NSJSONSerialization JSONObjectWithData:scopeData options:0 error:nil] : nil;
    if (![scope isKindOfClass:[NSDictionary class]]) return @{ @"status": @"invalid_scope" };
    NSDictionary *eventScope = scopeMap(scope[@"events"] ?: @[]);
    NSDictionary *reminderScope = scopeMap(scope[@"reminders"] ?: @[]);
    if ((eventScope.count && !readable(EKEntityTypeEvent)) ||
        (reminderScope.count && !readable(EKEntityTypeReminder)))
        return @{ @"status": @"authorization_required" };
    NSMutableArray *eventCalendars = [NSMutableArray array], *reminderCalendars = [NSMutableArray array];
    if (readable(EKEntityTypeEvent))
        for (EKCalendar *item in [store calendarsForEntityType:EKEntityTypeEvent])
            if (eventScope[item.calendarIdentifier]) [eventCalendars addObject:item];
    if (readable(EKEntityTypeReminder))
        for (EKCalendar *item in [store calendarsForEntityType:EKEntityTypeReminder])
            if (reminderScope[item.calendarIdentifier]) [reminderCalendars addObject:item];
    NSDate *now = [NSDate date];
    NSDate *from = [[NSCalendar currentCalendar] dateByAddingUnit:NSCalendarUnitDay value:-365 toDate:now options:0];
    NSDate *until = [[NSCalendar currentCalendar] dateByAddingUnit:NSCalendarUnitDay value:30 toDate:now options:0];
    NSMutableArray *events = [NSMutableArray array], *reminders = [NSMutableArray array];
    if (eventCalendars.count) {
        NSPredicate *predicate = [store predicateForEventsWithStartDate:from endDate:until calendars:eventCalendars];
        for (EKEvent *event in [store eventsMatchingPredicate:predicate]) {
            BOOL titleAllowed = [eventScope[event.calendar.calendarIdentifier] boolValue];
            NSString *title = titleAllowed ? [event.title substringToIndex:MIN(event.title.length, 160)] : @"";
            [events addObject:@{ @"id": event.calendarItemIdentifier ?: @"",
                                 @"calendarId": event.calendar.calendarIdentifier ?: @"",
                                 @"start": [formatter stringFromDate:event.startDate],
                                 @"end": [formatter stringFromDate:event.endDate],
                                 @"allDay": @(event.isAllDay), @"title": title ?: @"" }];
        }
    }
    if (reminderCalendars.count) {
        NSPredicate *predicate = [store predicateForRemindersInCalendars:reminderCalendars];
        __block NSArray<EKReminder *> *fetched = @[];
        __block BOOL reminderDone = NO;
        [store fetchRemindersMatchingPredicate:predicate completion:^(NSArray<EKReminder *> *items) {
            fetched = items ?: @[]; reminderDone = YES;
        }];
        waitUntil(^{ return reminderDone; }, 60);
        for (EKReminder *item in fetched) {
            NSDate *at = item.completionDate ?: [[NSCalendar currentCalendar] dateFromComponents:item.dueDateComponents];
            if (!at || [at compare:from] == NSOrderedAscending || [at compare:until] == NSOrderedDescending) continue;
            BOOL titleAllowed = [reminderScope[item.calendar.calendarIdentifier] boolValue];
            NSString *title = titleAllowed ? [item.title substringToIndex:MIN(item.title.length, 160)] : @"";
            [reminders addObject:@{ @"id": item.calendarItemIdentifier ?: @"",
                                    @"calendarId": item.calendar.calendarIdentifier ?: @"",
                                    @"at": [formatter stringFromDate:at],
                                    @"completed": @(item.isCompleted), @"title": title ?: @"" }];
        }
    }
    NSDictionary *payload = @{ @"generatedAt": [formatter stringFromDate:now],
                               @"events": events, @"reminders": reminders };
    NSData *bytes = [NSJSONSerialization dataWithJSONObject:payload options:NSJSONWritingSortedKeys error:nil];
    if (!bytes) return @{ @"status": @"encode_failed" };
    [[NSFileManager defaultManager] createDirectoryAtPath:[outputPath stringByDeletingLastPathComponent]
                              withIntermediateDirectories:YES attributes:@{ NSFilePosixPermissions: @0700 } error:nil];
    if (![bytes writeToFile:outputPath options:NSDataWritingAtomic error:nil])
        return @{ @"status": @"write_failed" };
    chmod(outputPath.fileSystemRepresentation, 0600);
    return @{ @"status": @"exported", @"eventCount": @(events.count),
              @"reminderCount": @(reminders.count) };
}
int main(int argc, const char **argv) {
    @autoreleasepool {
        umask(077);
        store = [EKEventStore new];
        formatter = [NSISO8601DateFormatter new];
        formatter.formatOptions = NSISO8601DateFormatWithInternetDateTime | NSISO8601DateFormatWithFractionalSeconds;
        NSArray<NSString *> *args = [[NSProcessInfo processInfo] arguments];
        if ([args containsObject:@"--status"]) {
            output(@{ @"eventsStatus": statusFor(EKEntityTypeEvent),
                      @"remindersStatus": statusFor(EKEntityTypeReminder) });
        } else if ([args containsObject:@"--request"]) {
            output(requestAccess());
        } else if ([args containsObject:@"--list"]) {
            output(listCalendars());
        } else if ([args containsObject:@"--scope"] && [args containsObject:@"--output"]) {
            NSUInteger scopeIndex = [args indexOfObject:@"--scope"], outputIndex = [args indexOfObject:@"--output"];
            if (scopeIndex+1 >= args.count || outputIndex+1 >= args.count) return 2;
            NSDictionary *result = exportScope(args[scopeIndex+1], args[outputIndex+1]);
            output(result);
            if (![result[@"status"] isEqualToString:@"exported"]) return 1;
        } else {
            output(@{ @"status": @"usage_error" }); return 2;
        }
    }
    return 0;
}
