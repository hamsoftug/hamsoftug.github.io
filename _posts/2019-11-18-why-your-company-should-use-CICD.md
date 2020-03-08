---
layout: post
title: How is Software made ?
featured-img: process
twitterimage: /assets/img/posts/process.jpg
image: /assets/img/posts/process.jpg
summary: # A tour of the Hamsoft Uganda software factory production line!
categories: [Technology,Software, Scrum,Design, Development, Agile]
keywords:
  - software development
  - software lifecycle
  - engineer software
---


CI and CD are two acronyms that are often mentioned when people talk about modern development practices. CI is straightforward and stands for continuous integration, a practice that focuses on making preparing a release easier. But CD can either mean continuous delivery or continuous deployment, and while those two practices have a lot in common, they also have a significant difference that can have critical consequences for a business.

We will see in this article what these three practices mean and what's required to use them.

## What are the differences between continuous integration, continuous delivery, and continuous deployment?

### Continuous integration

Developers practicing continuous integration merge their changes back to the main branch as often as possible. The developer's changes are validated by creating a build and running automated tests against the build. By doing so, you avoid the integration hell that usually happens when people wait for release day to merge their changes into the release branch.

Continuous integration puts a great emphasis on testing automation to check that the application is not broken whenever new commits are integrated into the main branch.

### Continuous delivery

[Continuous delivery](https://www.atlassian.com/continuous-delivery) is an extension of continuous integration to make sure that you can release new changes to your customers quickly in a sustainable way. This means that on top of having automated your testing, you also have automated your release process and you can deploy your application at any point of time by clicking on a button.

In theory, with continuous delivery, you can decide to release daily, weekly, fortnightly, or whatever suits your business requirements. However, if you truly want to get the benefits of continuous delivery, you should deploy to production as early as possible to make sure that you release small batches that are easy to troubleshoot in case of a problem.

### Continuous deployment

Continuous deployment goes one step further than continuous delivery. With this practice, every change that passes all stages of your production pipeline is released to your customers. There's no human intervention, and only a failed test will prevent a new change to be deployed to production.

Continuous deployment is an excellent way to accelerate the feedback loop with your customers and take pressure off the team as there isn't a *Release Day* anymore. Developers can focus on building software, and they see their work go live minutes after they've finished working on it.

### How the practices relate to each other

To put it simply continuous integration is part of both continuous delivery and continuous deployment. And continuous deployment is like continuous delivery, except that releases happen automatically.