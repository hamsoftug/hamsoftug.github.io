/**
 * Copyright 2016 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

// DO NOT EDIT THIS GENERATED OUTPUT DIRECTLY!
// This file should be overwritten as part of your build process.
// If you need to extend the behavior of the generated service worker, the best approach is to write
// additional code and include it using the importScripts option:
//   https://github.com/GoogleChrome/sw-precache#importscripts-arraystring
//
// Alternatively, it's possible to make changes to the underlying template file and then use that as the
// new base for generating output, via the templateFilePath option:
//   https://github.com/GoogleChrome/sw-precache#templatefilepath-string
//
// If you go that route, make sure that whenever you update your sw-precache dependency, you reconcile any
// changes made to this original template file with your modified copy.

// This generated service worker JavaScript will precache your site's resources.
// The code needs to be saved in a .js file at the top-level of your site, and registered
// from your pages in order to be used. See
// https://github.com/googlechrome/sw-precache/blob/master/demo/app/js/service-worker-registration.js
// for an example of how you can register this script and handle various service worker events.

/* eslint-env worker, serviceworker */
/* eslint-disable indent, no-unused-vars, no-multiple-empty-lines, max-nested-callbacks, space-before-function-paren, quotes, comma-spacing */
'use strict';

var precacheConfig = [["/404.html","2e9fa3f48469cb8884779c1d20dcaed4"],["/about-me/index.html","c52f7dcd78367cc2269b1f958a75b956"],["/assets/css/main.css","6774677a686189d9bcc9cb1e874f3c9c"],["/assets/img/blog/background.jpg","9537c4e8071f4903374848ebb894ce28"],["/assets/img/favicons/android-chrome-192x192.png","918969224b888aa650c465107fe35e78"],["/assets/img/favicons/android-chrome-512x512.png","dd628dfa9b8c86399ebe7dac5a6a75c7"],["/assets/img/favicons/apple-touch-icon.png","f6e4532af262e4e5bc955729f2fe7721"],["/assets/img/favicons/favicon-16x16.png","4d9762aca849718bb48101797a942df8"],["/assets/img/favicons/favicon-32x32.png","23baa78c4945427e923fc58fd16b28e9"],["/assets/img/favicons/mstile-144x144.png","4b4239f190790c014a80f03fe1ac408f"],["/assets/img/favicons/mstile-150x150.png","aec77ea9669364e6247159d64f63a202"],["/assets/img/favicons/mstile-310x150.png","4bba0e2e9fded14205edb4fe1651ce81"],["/assets/img/favicons/mstile-310x310.png","449a08f221938ea98b49b54d59b5f86b"],["/assets/img/favicons/mstile-70x70.png","6258a14609b321dbc2d8e2460d90c477"],["/assets/img/favicons/safari-pinned-tab.svg","1b7e81a89fdbf1072dcffc1a1df62ed1"],["/assets/img/icons/icon-email.svg","c66ee6f637138ab9d58d2e17d58c0aa7"],["/assets/img/icons/icon-github.svg","1bb027109345a90a9eab1e929d8669c2"],["/assets/img/icons/icon-instagram.svg","341a67c538d67f9ce92005cf14255dc2"],["/assets/img/icons/icon-linkedin.svg","97ce31c8546f65bd1e25afbbf86ec1de"],["/assets/img/icons/icon-medium.svg","275c731330eaf0b58fc09d6043d10e4a"],["/assets/img/icons/icon-twitter.svg","30551913d5399d6520e8a74b6f1e23f0"],["/assets/img/logos/imperial.svg","c9d07ac40c8a1fc81ef563575e7692ce"],["/assets/img/logos/morganstanley.svg","09247d21ad7d5c30f68122376424d2ad"],["/assets/img/logos/traventia.png","77dcf8cad2e72cb9ff3fce5f9f33ce34"],["/assets/img/posts/data-science.jpg","6f38d3973065839e16f9b0e1119bd37d"],["/assets/img/posts/data-science_lg.jpg","ce888157c1bbc885a0eb61b8c91b9632"],["/assets/img/posts/data-science_md.jpg","edc2f44b5d2799ae36fa711ede789aef"],["/assets/img/posts/data-science_placehold.jpg","c477ee0f4a3e66229514bffc7557471e"],["/assets/img/posts/data-science_sm.jpg","02f9bbed875df39bf2ceb8533036efd1"],["/assets/img/posts/data-science_thumb.jpg","1ad9337329aa0c08f8ec58deba92ccb5"],["/assets/img/posts/data-science_thumb@2x.jpg","cce2447c675162b7874a5df3c4e2b883"],["/assets/img/posts/data-science_xs.jpg","4f78219b4718ba72d007ea1973d6b4cc"],["/assets/img/posts/wild-mushrooms.jpg","e045d7346d7478c86a3343940e3b3539"],["/assets/img/posts/wild-mushrooms_lg.jpg","dacda6ce4179e7bc22c889f99be8eef7"],["/assets/img/posts/wild-mushrooms_md.jpg","59f149f8d9061d01fd9eacce338c8b1c"],["/assets/img/posts/wild-mushrooms_placehold.jpg","e99afdfb803a6c45d08670152cf9f6d4"],["/assets/img/posts/wild-mushrooms_sm.jpg","eeeaf148bfa97d30ab6d5f27a0083d20"],["/assets/img/posts/wild-mushrooms_thumb.jpg","8b3b83600f46715206411a5ef9c9c5a8"],["/assets/img/posts/wild-mushrooms_thumb@2x.jpg","03978ec191f8cebeebd1da60c461c118"],["/assets/img/posts/wild-mushrooms_xs.jpg","873a2326262d0997dfee2facf2bb7f8a"],["/assets/img/posts_contents/einstein-quote.jpg","ea998007f7bf70dde084f884859016cd"],["/assets/img/posts_contents/mushrooms-by-cap-color.png","87d9f6f95eb86c2e93b7ad0bebe2fd5e"],["/assets/img/posts_contents/mushrooms-by-odor.png","38097c3acf6da98c0bcd960e9bd0ab51"],["/assets/img/posts_contents/mushrooms-by-spore-print-color.png","6ae2b93aaa56097a23482dc0167ff8e6"],["/assets/img/posts_contents/thats-all-folks.jpg","6a4e0de988bfc6b8fd22bd0fff953f48"],["/assets/js/bundle.js","1815b7ec0022f5855ae9171e9f3c5233"],["/blog/determining-the-edibility-of-wild-mushrooms/index.html","09d39b088bebcf773d0f679973a80b59"],["/blog/index.html","691bd1e0fdd6ae3718a93c876d467e67"],["/blog/why-I-am-starting-a-data-science-blog/index.html","a5a408591192a127e55ac192c6c41101"],["/categories/index.html","ea2a85c285a2355d61d314cdf4aaf0cc"],["/contact/index.html","0938dbcf0696e3d16a03ec1199709e50"],["/index.html","691904302b81f95d885dc1b597dfec3b"],["/sw.js","7eb85802917edf8228ebe9f98cf9b89a"]];
var cacheName = 'sw-precache-v3--' + (self.registration ? self.registration.scope : '');


var ignoreUrlParametersMatching = [/^utm_/];



var addDirectoryIndex = function (originalUrl, index) {
    var url = new URL(originalUrl);
    if (url.pathname.slice(-1) === '/') {
      url.pathname += index;
    }
    return url.toString();
  };

var cleanResponse = function (originalResponse) {
    // If this is not a redirected response, then we don't have to do anything.
    if (!originalResponse.redirected) {
      return Promise.resolve(originalResponse);
    }

    // Firefox 50 and below doesn't support the Response.body stream, so we may
    // need to read the entire body to memory as a Blob.
    var bodyPromise = 'body' in originalResponse ?
      Promise.resolve(originalResponse.body) :
      originalResponse.blob();

    return bodyPromise.then(function(body) {
      // new Response() is happy when passed either a stream or a Blob.
      return new Response(body, {
        headers: originalResponse.headers,
        status: originalResponse.status,
        statusText: originalResponse.statusText
      });
    });
  };

var createCacheKey = function (originalUrl, paramName, paramValue,
                           dontCacheBustUrlsMatching) {
    // Create a new URL object to avoid modifying originalUrl.
    var url = new URL(originalUrl);

    // If dontCacheBustUrlsMatching is not set, or if we don't have a match,
    // then add in the extra cache-busting URL parameter.
    if (!dontCacheBustUrlsMatching ||
        !(url.pathname.match(dontCacheBustUrlsMatching))) {
      url.search += (url.search ? '&' : '') +
        encodeURIComponent(paramName) + '=' + encodeURIComponent(paramValue);
    }

    return url.toString();
  };

var isPathWhitelisted = function (whitelist, absoluteUrlString) {
    // If the whitelist is empty, then consider all URLs to be whitelisted.
    if (whitelist.length === 0) {
      return true;
    }

    // Otherwise compare each path regex to the path of the URL passed in.
    var path = (new URL(absoluteUrlString)).pathname;
    return whitelist.some(function(whitelistedPathRegex) {
      return path.match(whitelistedPathRegex);
    });
  };

var stripIgnoredUrlParameters = function (originalUrl,
    ignoreUrlParametersMatching) {
    var url = new URL(originalUrl);
    // Remove the hash; see https://github.com/GoogleChrome/sw-precache/issues/290
    url.hash = '';

    url.search = url.search.slice(1) // Exclude initial '?'
      .split('&') // Split into an array of 'key=value' strings
      .map(function(kv) {
        return kv.split('='); // Split each 'key=value' string into a [key, value] array
      })
      .filter(function(kv) {
        return ignoreUrlParametersMatching.every(function(ignoredRegex) {
          return !ignoredRegex.test(kv[0]); // Return true iff the key doesn't match any of the regexes.
        });
      })
      .map(function(kv) {
        return kv.join('='); // Join each [key, value] array into a 'key=value' string
      })
      .join('&'); // Join the array of 'key=value' strings into a string with '&' in between each

    return url.toString();
  };


var hashParamName = '_sw-precache';
var urlsToCacheKeys = new Map(
  precacheConfig.map(function(item) {
    var relativeUrl = item[0];
    var hash = item[1];
    var absoluteUrl = new URL(relativeUrl, self.location);
    var cacheKey = createCacheKey(absoluteUrl, hashParamName, hash, false);
    return [absoluteUrl.toString(), cacheKey];
  })
);

function setOfCachedUrls(cache) {
  return cache.keys().then(function(requests) {
    return requests.map(function(request) {
      return request.url;
    });
  }).then(function(urls) {
    return new Set(urls);
  });
}

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return setOfCachedUrls(cache).then(function(cachedUrls) {
        return Promise.all(
          Array.from(urlsToCacheKeys.values()).map(function(cacheKey) {
            // If we don't have a key matching url in the cache already, add it.
            if (!cachedUrls.has(cacheKey)) {
              var request = new Request(cacheKey, {credentials: 'same-origin'});
              return fetch(request).then(function(response) {
                // Bail out of installation unless we get back a 200 OK for
                // every request.
                if (!response.ok) {
                  throw new Error('Request for ' + cacheKey + ' returned a ' +
                    'response with status ' + response.status);
                }

                return cleanResponse(response).then(function(responseToCache) {
                  return cache.put(cacheKey, responseToCache);
                });
              });
            }
          })
        );
      });
    }).then(function() {
      
      // Force the SW to transition from installing -> active state
      return self.skipWaiting();
      
    })
  );
});

self.addEventListener('activate', function(event) {
  var setOfExpectedUrls = new Set(urlsToCacheKeys.values());

  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return cache.keys().then(function(existingRequests) {
        return Promise.all(
          existingRequests.map(function(existingRequest) {
            if (!setOfExpectedUrls.has(existingRequest.url)) {
              return cache.delete(existingRequest);
            }
          })
        );
      });
    }).then(function() {
      
      return self.clients.claim();
      
    })
  );
});


self.addEventListener('fetch', function(event) {
  if (event.request.method === 'GET') {
    // Should we call event.respondWith() inside this fetch event handler?
    // This needs to be determined synchronously, which will give other fetch
    // handlers a chance to handle the request if need be.
    var shouldRespond;

    // First, remove all the ignored parameters and hash fragment, and see if we
    // have that URL in our cache. If so, great! shouldRespond will be true.
    var url = stripIgnoredUrlParameters(event.request.url, ignoreUrlParametersMatching);
    shouldRespond = urlsToCacheKeys.has(url);

    // If shouldRespond is false, check again, this time with 'index.html'
    // (or whatever the directoryIndex option is set to) at the end.
    var directoryIndex = 'index.html';
    if (!shouldRespond && directoryIndex) {
      url = addDirectoryIndex(url, directoryIndex);
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond is still false, check to see if this is a navigation
    // request, and if so, whether the URL matches navigateFallbackWhitelist.
    var navigateFallback = '';
    if (!shouldRespond &&
        navigateFallback &&
        (event.request.mode === 'navigate') &&
        isPathWhitelisted([], event.request.url)) {
      url = new URL(navigateFallback, self.location).toString();
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond was set to true at any point, then call
    // event.respondWith(), using the appropriate cache key.
    if (shouldRespond) {
      event.respondWith(
        caches.open(cacheName).then(function(cache) {
          return cache.match(urlsToCacheKeys.get(url)).then(function(response) {
            if (response) {
              return response;
            }
            throw Error('The cached response that was expected is missing.');
          });
        }).catch(function(e) {
          // Fall back to just fetch()ing the request if some unexpected error
          // prevented the cached response from being valid.
          console.warn('Couldn\'t serve response for "%s" from cache: %O', event.request.url, e);
          return fetch(event.request);
        })
      );
    }
  }
});







