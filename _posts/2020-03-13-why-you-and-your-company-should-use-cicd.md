---
layout: post
title: Here is why your company should use CICD in 
featured-img: cicd
twitterimage: /assets/img/posts/cicd.jpg
image: /assets/img/posts/cicd.jpg
summary: # CI/CD Together, these tools increase code quality, reduce development time, easily reveal errors and will generally make your life easier.
categories: [Technology,Software, Development, Agile]
keywords:
  - continuous integration
  - continous deployment 
  - CI/CD
---

### Why do I need CI/CD ?

You and I remember that Friday morning when you  checked out master to see what was in there , then you forgot to switch back to your development branch.  You embarked on your days task,  worked the whole day writing code nd as  evening approached you realized that the feature couldn't be completed that day and so you didn't see the need to write tests for an incomplete feature ,  you committed , pushed and closed your laptop and headed for home.

 With the only thing in mind being to go home,  your memory couldn't serve you well to remind you that the whole day you had been working and even pushed  code directly to master. 

#### Trouble sets in

You reached home and after a cold shower, you are ready to get up to speed with latest around your family , friends and of course news then all of a  sudden  you receive a call from your project manager saying that the system is down, and that you need to take action right away or else your company might lose its reputation to the complaining customers..

![image for trouble](https://blog.hamsoftug.com/assets/img/posts_contents/sebastian-herrmannh.jpg)

Immediately you reach out for your laptop from the bedroom , with little  suspicion you check your email for any error reports. from the error reports it takes to code ,you quickly review you and see that it is your last commit wrecking havoc.

Ohhh snap it dawns on you that you actually pushed unfinished code into master . as a colleague deployed ,everything went along with your buggy code to the production.

#### How would CI/CD have saved me ?

Allow me to quickly to take this out of the way by saying that CI/CD together increase code quality, reduce development time, easily reveal errors and will generally make your life easier.

Okay so to address the question at  hand here is how 

Now that you have learned from experience you can now proceed to the technical aspects of what really CI/CD is.

### So what is continuous integration, continuous delivery, and continuous deployment?

CI and CD are two acronyms that are often mentioned when people talk about modern software development practices. CI is straightforward and stands for continuous integration, a practice that focuses on making preparing a release easier(code reviews,tests ,code quality and coding standards ) are enforce from here. But CD can either mean continuous delivery or continuous deployment, and while those two practices have a lot in common, they also have a significant difference that can have critical consequences for a business.

#### Continuous integration

Developers practicing continuous integration merge their changes back to the main branch as often as possible. The developer's changes are validated by creating a build and running automated tests against the build. By doing so, you avoid the integration hell that usually happens when people wait for release day to merge their changes into the release branch.

Continuous integration puts a great emphasis on testing automation to check that the application is not broken whenever new commits are integrated into the main branch.

#### Continuous delivery

Continuous delivery is an extension of continuous integration to make sure that you can release new changes to your customers quickly in a sustainable way. This means that on top of having automated your testing, you also have automated your release process and you can deploy your application at any point of time by clicking on a button.

In theory, with continuous delivery, you can decide to release daily, weekly, fortnightly, or whatever suits your business requirements. However, if you truly want to get the benefits of continuous delivery, you should deploy to production as early as possible to make sure that you release small batches that are easy to troubleshoot in case of a problem.[source](https://www.atlassian.com/continuous-delivery)

#### Continuous deployment

Continuous deployment goes one step further than continuous delivery. With this practice, every change that passes all stages of your production pipeline is released to your customers. There's no human intervention, and only a failed test will prevent a new change to be deployed to production.

Continuous deployment is an excellent way to accelerate the feedback loop with your customers and take pressure off the team as there isn't a *Release Day* anymore. Developers can focus on building software, and they see their work go live minutes after they've finished working on it. [source](bitbucket.org)

### Final thoughts

To put it simply ,if you wan to ship software as a pro and "write silicon valley grade code"(is there anything like this?) then you are too late to start using ci/cd in your workflows. use some of the available solutions such as GitHub actions,Bitbucket pipeline etc..

![](https://blog.hamsoftug.com/assets/img/posts_contents/thats-all-folks.jpg)